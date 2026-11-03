from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
from uuid import uuid4

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from federated import IMAGE_SUFFIXES, IMAGENET_TRANSFORM, MAX_DATASET_FILES, MAX_DATASET_UPLOAD_BYTES, MAX_IMAGE_PIXELS, create_densenet, local_train, validate_dataset, weighted_fedavg
from model_transport import manifest_from_payload, pack_state_dict, unpack_state_dict
from medical_tasks import TASKS, get_task, task_checkpoint

app = FastAPI(title='ArogyaVaani Local Training Service')
LOCAL_SERVICE_TOKEN = os.environ.get('LOCAL_SERVICE_TOKEN', '')
LOCAL_DATA_ROOT = Path(os.environ.get('LOCAL_DATA_ROOT', 'local_datasets')).expanduser().resolve()
LOCAL_UPDATE_ROOT = Path(os.environ.get('LOCAL_UPDATE_ROOT', 'local_updates')).expanduser().resolve()
MAX_PREDICTION_BYTES = int(os.environ.get('MAX_PREDICTION_BYTES', str(50 * 1024 * 1024)))
# Folder selection is a simple multipart request, but training and model
# submission send JSON and therefore require a CORS preflight. Vite may move
# to another local port when 5173 is occupied; permit loopback development
# origins without opening the local trainer to arbitrary network hosts.
allowed_origins = [origin.strip().rstrip('/') for origin in os.environ.get('LOCAL_FRONTEND_ORIGIN', '').split(',') if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ['*'],
    allow_origin_regex=r'^(https?://(localhost|127\.0\.0\.1)(?::\d+)?|https://.*\.vercel\.app)$',
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization', 'X-ArogyaVaani-Local-Token', 'Access-Control-Request-Private-Network', '*'],
)
training_status: dict = {'status': 'idle'}
training_jobs: dict[str, dict] = {}
latest_training_job: str | None = None
training_jobs_lock = threading.Lock()
DATASET_UPLOAD_ROOT = LOCAL_DATA_ROOT


@app.middleware('http')
async def local_service_auth(request: Request, call_next):
    """Optional shared local-network token; deployments should set it when the UI is remote."""
    if LOCAL_SERVICE_TOKEN and request.url.path not in {'/health', '/ready'}:
        supplied = request.headers.get('X-ArogyaVaani-Local-Token', '')
        if not hmac.compare_digest(supplied, LOCAL_SERVICE_TOKEN):
            raise HTTPException(401, 'Local service authentication is required')
    # Reject declared oversized local uploads before multipart parsing. The
    # endpoint validators retain the exact file/dataset limits as a second
    # line of defense for chunked requests without Content-Length.
    declared_length = request.headers.get('content-length')
    limit = MAX_DATASET_UPLOAD_BYTES + 10 * 1024 * 1024 if request.url.path == '/dataset/upload' else MAX_PREDICTION_BYTES if request.url.path == '/predict' else None
    if limit is not None and declared_length and declared_length.isdigit() and int(declared_length) > limit:
        raise HTTPException(413, 'Local upload exceeds the configured size limit')
    response = await call_next(request)
    if request.headers.get('access-control-request-private-network') == 'true':
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response


def checkpoint_for_task(task_id: str, requested: str | None, name: str) -> Path:
    """Keep checkpoints for one task from overwriting another task's model."""
    default = Path(task_checkpoint(task_id, name))
    if requested is None:
        return default
    candidate = Path(requested).resolve()
    task_root = default.parent.resolve()
    try:
        candidate.relative_to(task_root)
    except ValueError as error:
        raise ValueError('Checkpoint paths must stay inside the selected task local-model directory') from error
    return candidate


def dataset_for_task(task_id: str, requested: str) -> Path:
    """Accept only paths created by this local service for the selected task."""
    candidate = Path(requested).expanduser().resolve()
    task_root = (DATASET_UPLOAD_ROOT / get_task(task_id).task_id.lower() / 'uploads').resolve()
    try:
        candidate.relative_to(task_root)
    except ValueError as error:
        raise ValueError('Dataset paths must be created inside the selected task local dataset directory') from error
    return candidate


def validate_checkpoint_metadata(saved: dict, task_id: str) -> list[str]:
    task = get_task(task_id)
    if saved.get('task') != task.task_id or saved.get('architecture', task.architecture) != task.architecture:
        raise ValueError('The local checkpoint does not match the selected medical task')
    classes = saved.get('classes')
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(label, str) and label for label in classes):
        raise ValueError('The local checkpoint has invalid class labels')
    if task.class_labels and tuple(c.upper() for c in classes) != tuple(c.upper() for c in task.class_labels):
        raise ValueError(f'The local checkpoint class labels {classes} do not match the selected medical task {list(task.class_labels)}')
    if not isinstance(saved.get('state_dict'), dict):
        raise ValueError('The local checkpoint has no model state')
    return classes


def load_inference_checkpoint(checkpoint: Path, task_id: str) -> tuple[dict, list[str]] | None:
    """Load a usable checkpoint, returning None for an absent or stale candidate."""
    if not checkpoint.is_file():
        return None
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        return saved, validate_checkpoint_metadata(saved, task_id)
    except (OSError, RuntimeError, KeyError, ValueError):
        return None


def select_inference_checkpoint(task_id: str, checkpoint_path: str | None = None) -> tuple[Path, dict, list[str]]:
    """Choose a trained global model, or local training while global is bootstrap v0.

    Version zero is only the shared random bootstrap used to begin federated
    training. It must not mask a locally trained checkpoint before an
    aggregated global model is available.
    """
    if checkpoint_path is not None:
        requested = checkpoint_for_task(task_id, checkpoint_path, 'global.pt')
        loaded = load_inference_checkpoint(requested, task_id)
        if loaded is None:
            raise ValueError('The requested local model checkpoint is unavailable or invalid')
        saved, classes = loaded
        return requested, saved, classes

    global_checkpoint = checkpoint_for_task(task_id, None, 'global.pt')
    local_checkpoint = checkpoint_for_task(task_id, None, 'latest.pt')
    global_model = load_inference_checkpoint(global_checkpoint, task_id)
    if global_model is not None:
        saved, classes = global_model
        try:
            global_version = int(saved.get('version', 0))
        except (TypeError, ValueError):
            global_version = 0
        if global_version > 0:
            return global_checkpoint, saved, classes

    local_model = load_inference_checkpoint(local_checkpoint, task_id)
    if local_model is not None:
        saved, classes = local_model
        return local_checkpoint, saved, classes
    if global_model is not None:
        saved, classes = global_model
        return global_checkpoint, saved, classes
    raise ValueError('No synchronized global or local trained model checkpoint is available')


def initialize_bootstrap_model(task_id: str, classes: list[str], checkpoint_path: Path, overwrite: bool = False) -> None:
    """Persist the deterministic, shared v0 model for a hospital-local task."""
    if checkpoint_path.is_file() and not overwrite:
        return
    task = get_task(task_id)
    model = create_densenet(len(classes))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'task': task.task_id, 'state_dict': model.state_dict(), 'classes': classes,
        'architecture': task.architecture, 'version': 0, 'bootstrap': True,
        'synchronized_at': datetime.now(timezone.utc).isoformat(),
    }, checkpoint_path)

class Update(BaseModel):
    sample_count: int = Field(gt=0)
    parameters: dict[str, list[float]]

class TrainingRequest(BaseModel):
    task: str = 'PNEUMONIA'
    dataset_dir: str
    num_classes: int = Field(gt=1)
    epochs: int = Field(default=10, ge=1, le=100)
    checkpoint_path: str | None = None
    global_checkpoint_path: str | None = None
    # Federation context is captured with the job, rather than inferred from a
    # browser page after training has completed.  The bearer token is used only
    # while this running job attempts its first delivery and is never written
    # to hospital-local storage.
    federation_round: int | None = Field(default=None, gt=0)
    base_global_model_version: int | None = Field(default=None, ge=0)
    federation_api_url: str | None = None
    federation_token: str | None = None

class SynchronizeRequest(BaseModel):
    task: str = 'PNEUMONIA'
    checkpoint_path: str | None = None
    state_dict: dict[str, list[float]]
    parameter_shapes: dict[str, list[int]] = {}
    classes: list[str] = []
    version: int = Field(ge=0)

class SubmitUpdateRequest(BaseModel):
    task: str = 'PNEUMONIA'
    api_url: str
    token: str = Field(min_length=1)
    round: int = Field(gt=0)
    checkpoint_path: str | None = None

class RetryUpdateRequest(BaseModel):
    task: str = 'PNEUMONIA'
    round: int = Field(gt=0)
    api_url: str
    token: str = Field(min_length=1)

class DownloadGlobalRequest(BaseModel):
    task: str = 'PNEUMONIA'
    api_url: str
    token: str = Field(min_length=1)
    checkpoint_path: str | None = None
    expected_version: int | None = Field(default=None, ge=0)


class BootstrapModelRequest(BaseModel):
    task: str = 'PNEUMONIA'
    classes: list[str] = Field(min_length=2, max_length=1_000)

class ModelUpdateRequest(BaseModel):
    task: str = 'PNEUMONIA'
    checkpoint_path: str | None = None


def training_jobs_path() -> Path:
    return LOCAL_UPDATE_ROOT / 'training-jobs.json'


def pending_update_path(task_id: str, round_number: int) -> Path:
    """AVM1 updates stay hospital-local until the coordinator accepts them."""
    get_task(task_id)
    if round_number <= 0:
        raise ValueError('A valid federation round is required')
    return LOCAL_UPDATE_ROOT / 'pending' / task_id.lower() / f'round-{round_number}.avm1'


def pending_update_metadata_path(task_id: str, round_number: int) -> Path:
    return pending_update_path(task_id, round_number).with_suffix('.json')


def persist_pending_update(task_id: str, round_number: int, payload: bytes, metadata: dict) -> None:
    destination = pending_update_path(task_id, round_number)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(destination)
    metadata = {**metadata, 'task': task_id, 'round': round_number, 'payload_bytes': len(payload), 'checksum': hashlib.sha256(payload).hexdigest(), 'status': 'pending', 'updated_at': datetime.now(timezone.utc).isoformat()}
    metadata_path = pending_update_metadata_path(task_id, round_number)
    metadata_temporary = metadata_path.with_suffix('.tmp')
    metadata_temporary.write_text(json.dumps(metadata), encoding='utf-8')
    metadata_temporary.replace(metadata_path)


def read_pending_update(task_id: str, round_number: int) -> tuple[bytes, dict]:
    payload_path = pending_update_path(task_id, round_number)
    metadata_path = pending_update_metadata_path(task_id, round_number)
    if not payload_path.is_file() or not metadata_path.is_file():
        raise ValueError('No pending local model update exists for this task round')
    payload, metadata = payload_path.read_bytes(), json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('task') != task_id or metadata.get('round') != round_number:
        raise ValueError('Pending update metadata is invalid')
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), str(metadata.get('checksum', ''))):
        raise ValueError('Pending update integrity check failed')
    # Re-parse before each transmission, so a corrupted local artifact never
    # leaves the hospital even if its sidecar file was altered.
    manifest = manifest_from_payload(payload)
    if manifest.get('task') != task_id:
        raise ValueError('Pending update belongs to a different medical task')
    return payload, metadata


def mark_pending_update(task_id: str, round_number: int, **changes: object) -> None:
    payload, metadata = read_pending_update(task_id, round_number)
    persist_pending_update(task_id, round_number, payload, {**metadata, **changes})


def delete_pending_update(task_id: str, round_number: int) -> None:
    # A local update is removed only after the coordinator has sent a successful
    # response.  Missing files are harmless during a recovery/retry race.
    pending_update_path(task_id, round_number).unlink(missing_ok=True)
    pending_update_metadata_path(task_id, round_number).unlink(missing_ok=True)


def submit_pending_update(task_id: str, round_number: int, api_url: str, token: str) -> dict:
    payload, metadata = read_pending_update(task_id, round_number)
    endpoint = api_url.rstrip('/') + '/api/federated/rounds/' + str(round_number) + '/updates?samples=' + str(metadata['samples']) + '&task=' + task_id
    try:
        response = urlopen(UrlRequest(endpoint, data=payload, method='POST', headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
        response_body = json.loads(response.read())
    except (URLError, OSError) as error:
        if 'localhost' in endpoint:
            try:
                fallback_endpoint = endpoint.replace('://localhost:', '://127.0.0.1:')
                response = urlopen(UrlRequest(fallback_endpoint, data=payload, method='POST', headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
                response_body = json.loads(response.read())
                delete_pending_update(task_id, round_number)
                return {'submitted': True, **response_body, 'bytes': len(payload)}
            except Exception:
                pass
        detail = error.read().decode(errors='replace') if isinstance(error, HTTPError) else str(error)
        mark_pending_update(task_id, round_number, last_error=detail[:1000], last_attempt_at=datetime.now(timezone.utc).isoformat())
        raise
    except (HTTPError, ValueError) as error:
        detail = error.read().decode(errors='replace') if isinstance(error, HTTPError) else str(error)
        mark_pending_update(task_id, round_number, last_error=detail[:1000], last_attempt_at=datetime.now(timezone.utc).isoformat())
        raise
    delete_pending_update(task_id, round_number)
    return {'submitted': True, **response_body, 'bytes': len(payload)}


def persist_training_jobs() -> None:
    """Persist non-sensitive job state so an interrupted worker is visible after restart."""
    try:
        LOCAL_UPDATE_ROOT.mkdir(parents=True, exist_ok=True)
        destination = training_jobs_path()
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(training_jobs), encoding='utf-8')
        temporary.replace(destination)
    except OSError:
        # Training itself will surface a storage error when it writes a model.
        # Do not make the health endpoint disclose filesystem details.
        return


def restore_training_jobs() -> None:
    global latest_training_job, training_status
    try:
        raw = json.loads(training_jobs_path().read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            return
        training_jobs.update({key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, dict)})
        for job in training_jobs.values():
            if job.get('status') in {'queued', 'validating', 'training', 'packaging', 'submitting'}:
                job.update({'status': 'failed', 'message': 'Local training was interrupted or cancelled.', 'updated_at': datetime.now(timezone.utc).isoformat()})
        if training_jobs:
            latest_training_job = max(training_jobs, key=lambda key: str(training_jobs[key].get('updated_at', '')))
            training_status = dict(training_jobs[latest_training_job])
            persist_training_jobs()
    except (OSError, ValueError, json.JSONDecodeError):
        return

try:
    restore_training_jobs()
except Exception:
    pass


def update_training_job(job_id: str, **changes: object) -> dict:
    global training_status
    with training_jobs_lock:
        job = training_jobs.setdefault(job_id, {'job_id': job_id})
        job.update(changes)
        job['updated_at'] = datetime.now(timezone.utc).isoformat()
        training_status = dict(job)
        persist_training_jobs()
        return dict(job)


def record_central_training_state(request: TrainingRequest, state: dict) -> None:
    """Best-effort operational metadata only; images and artifacts stay local."""
    if not request.federation_api_url or not request.federation_token:
        return
    payload = {key: state[key] for key in ('task', 'status', 'samples', 'epochs', 'loss', 'accuracy', 'message', 'created_at', 'updated_at') if key in state}
    payload['task'] = request.task
    if state.get('status') in {'completed', 'failed'}:
        payload['completedAt'] = datetime.now(timezone.utc).isoformat()
    try:
        body = json.dumps(payload).encode('utf-8')
        urlopen(UrlRequest(request.federation_api_url.rstrip('/') + '/api/training/runs', data=body, method='POST', headers={'Authorization': 'Bearer ' + request.federation_token, 'Content-Type': 'application/json'}), timeout=15).read()
    except (HTTPError, URLError, OSError, ValueError):
        # Delivery failures do not affect local model durability or training.
        return

@app.get('/health')
def health():
    return {'status': 'operational', 'privacy': 'images stay local', 'tasks': [task.task_id for task in TASKS.values() if task.enabled]}


@app.get('/ready')
def ready():
    try:
        DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        Path(task_checkpoint('PNEUMONIA', 'ready-check')).parent.mkdir(parents=True, exist_ok=True)
        LOCAL_UPDATE_ROOT.mkdir(parents=True, exist_ok=True)
        return {'status': 'ready', 'storage': 'available'}
    except OSError:
        raise HTTPException(503, 'Local storage is unavailable')


@app.on_event('startup')
def initialize_local_service() -> None:
    restore_training_jobs()

@app.get('/medical-tasks')
def list_medical_tasks():
    return [task.__dict__ for task in TASKS.values() if task.enabled]

@app.get('/dataset/validate')
def dataset_validation(dataset_dir: str, task: str = 'PNEUMONIA'):
    try:
        return validate_dataset(str(dataset_for_task(task, dataset_dir)))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

def safe_relative_path(name: str) -> Path:
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or not path.parts or any(part in {'', '.', '..'} or ':' in part for part in path.parts):
        raise ValueError('Dataset contains an unsafe file path')
    return Path(*path.parts)

def folder_upload_path(name: str) -> tuple[str, Path]:
    """Return the selected dataset folder name and its path below that folder.

    Chromium's directory picker supplies names such as
    ``chest_xray/NORMAL/image1.jpeg``.  The selected parent folder is removed
    only after validation, so the local service reconstructs
    ``dataset/NORMAL/image1.jpeg`` for ImageFolder.
    """
    path = safe_relative_path(name)
    if len(path.parts) < 3:
        raise ValueError('Select the parent dataset folder containing class folders, not individual files')
    return path.parts[0], Path(*path.parts[1:])

async def save_upload(upload: UploadFile, destination: Path, total_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as target:
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_DATASET_UPLOAD_BYTES:
                raise ValueError('Dataset upload exceeds the 1 GB local limit')
            target.write(chunk)
    return total_bytes

@app.post('/dataset/upload')
async def upload_dataset(task: str = 'PNEUMONIA', files: list[UploadFile] = File(...)):
    """Receive a browser-selected dataset folder into hospital-local storage only."""
    if not files:
        raise HTTPException(400, 'Select a dataset folder')
    if len(files) > MAX_DATASET_FILES:
        raise HTTPException(400, f'Dataset contains too many files (maximum {MAX_DATASET_FILES:,})')
    try:
        task_config = get_task(task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    upload_dir = DATASET_UPLOAD_ROOT / task_config.task_id.lower() / 'uploads' / uuid4().hex
    dataset_dir = upload_dir / 'dataset'
    total_bytes = 0
    try:
        DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        dataset_name = ''
        stored_paths: set[Path] = set()
        for upload in files:
            selected_folder, relative = folder_upload_path(upload.filename or '')
            if not dataset_name:
                dataset_name = selected_folder
            elif selected_folder != dataset_name:
                raise ValueError('All files must belong to one selected dataset folder')
            if relative.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError('Dataset folders may contain only JPG, JPEG, PNG, BMP, TIFF, or WEBP images')
            if relative in stored_paths:
                raise ValueError('Dataset contains duplicate relative file paths')
            stored_paths.add(relative)
            total_bytes = await save_upload(upload, dataset_dir / relative, total_bytes)
        validation = validate_dataset(str(dataset_dir))
        return {'task': task_config.task_id, 'dataset_dir': str(dataset_dir), 'dataset_name': dataset_name, 'validation': validation, 'storage': 'local-only'}
    except (OSError, ValueError) as error:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(400, str(error) if str(error) else 'Dataset upload could not be processed') from error

@app.get('/training/status')
def get_training_status():
    return training_status


@app.get('/training/status/{job_id}')
def get_training_job_status(job_id: str):
    job = training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'Local training job was not found')
    return job


@app.get('/training/jobs')
def list_training_jobs(task: str | None = None):
    try:
        if task:
            get_task(task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    jobs = [dict(job) for job in training_jobs.values() if not task or job.get('task') == task]
    return sorted(jobs, key=lambda job: str(job.get('updated_at', job.get('created_at', ''))), reverse=True)


@app.get('/local-state')
def local_state(task: str = 'PNEUMONIA'):
    try:
        task_config = get_task(task)
        jobs = [dict(job) for job in training_jobs.values() if job.get('task') == task_config.task_id]
        latest = max(jobs, key=lambda job: str(job.get('updated_at', '')), default=None)
        checkpoint = checkpoint_for_task(task_config.task_id, None, 'latest.pt')
        global_checkpoint = checkpoint_for_task(task_config.task_id, None, 'global.pt')
        pending = []
        pending_root = LOCAL_UPDATE_ROOT / 'pending' / task_config.task_id.lower()
        if pending_root.is_dir():
            for item in pending_root.glob('round-*.json'):
                try:
                    metadata = json.loads(item.read_text(encoding='utf-8'))
                    pending.append({key: metadata.get(key) for key in ('task', 'round', 'status', 'samples', 'base_global_model_version', 'created_at', 'updated_at', 'last_error')})
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        return {'task': task_config.task_id, 'latest_training_job': latest, 'latest_checkpoint': str(checkpoint) if checkpoint.is_file() else None, 'global_checkpoint': str(global_checkpoint) if global_checkpoint.is_file() else None, 'pending_updates': sorted(pending, key=lambda item: item.get('round', 0), reverse=True)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

def run_training(job_id: str, request: TrainingRequest):
    try:
        task = get_task(request.task)
        record_central_training_state(request, update_training_job(job_id, status='validating', task=task.task_id, epochs=request.epochs))
        def update_progress(progress: dict):
            update_training_job(job_id, task=task.task_id, **progress)
        dataset_dir = dataset_for_task(task.task_id, request.dataset_dir)
        validation = validate_dataset(str(dataset_dir))
        if task.class_labels and tuple(c.upper() for c in validation['classes']) != tuple(c.upper() for c in task.class_labels):
            raise ValueError(f"Local dataset classes {validation['classes']} do not match expected classes for {task.display_name}: {list(task.class_labels)}")
        if request.num_classes != len(validation['classes']):
            raise ValueError('Requested output classes do not match the validated local dataset')
        global_checkpoint_path = checkpoint_for_task(task.task_id, request.global_checkpoint_path, 'global.pt')
        base_version = 0
        initial_checkpoint_to_use = None
        if Path(global_checkpoint_path).is_file():
            global_checkpoint = torch.load(global_checkpoint_path, map_location='cpu', weights_only=True)
            try:
                base_version = int(global_checkpoint.get('version', 0))
            except (ValueError, TypeError):
                base_version = 0
            saved_classes = global_checkpoint.get('classes', [])
            state_dict = global_checkpoint.get('state_dict') if isinstance(global_checkpoint, dict) else global_checkpoint
            classifier_weight = state_dict.get('classifier.weight') if isinstance(state_dict, dict) else None
            shape_mismatch = classifier_weight is not None and classifier_weight.shape[0] != len(validation['classes'])
            if saved_classes != validation['classes'] or shape_mismatch:
                if base_version == 0 or request.base_global_model_version == 0 or request.base_global_model_version is None:
                    initialize_bootstrap_model(task.task_id, validation['classes'], Path(global_checkpoint_path), overwrite=True)
                    base_version = 0
                    initial_checkpoint_to_use = str(global_checkpoint_path)
                else:
                    raise ValueError(f"Local global model classes ({saved_classes}) do not match dataset classes ({validation['classes']}). Please synchronize the active global model for {task.display_name}.")
            else:
                initial_checkpoint_to_use = str(global_checkpoint_path)
        elif request.base_global_model_version == 0 or request.base_global_model_version is None:
            initialize_bootstrap_model(task.task_id, validation['classes'], Path(global_checkpoint_path), overwrite=True)
            base_version = 0
            initial_checkpoint_to_use = str(global_checkpoint_path)
        if request.base_global_model_version is not None and base_version != request.base_global_model_version:
            raise ValueError(f'Local global model v{base_version} does not match required federation base v{request.base_global_model_version}')
        state, metrics = local_train(str(dataset_dir), request.num_classes, request.epochs, initial_checkpoint=initial_checkpoint_to_use, progress=update_progress)
        collapsed_classes = [label for label, values in metrics['class_accuracy'].items() if values['accuracy'] == 0]
        if collapsed_classes:
            raise ValueError(f"Training did not recognize any images in: {', '.join(collapsed_classes)}. Increase the training data or epochs and retrain; no model update was submitted.")
        checkpoint = checkpoint_for_task(task.task_id, request.checkpoint_path, 'latest.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = {'task': task.task_id, 'state_dict': state, 'classes': metrics['classes'], 'metrics': metrics, 'architecture': task.architecture, 'base_version': base_version, 'federation_round': request.federation_round, 'created_at': datetime.now(timezone.utc).isoformat()}
        torch.save(checkpoint_metadata, checkpoint)
        
        # If part of a federation round, transition through packaging and submitting
        # so clients monitoring training status see the accurate transfer lifecycle.
        if request.federation_round:
            update_training_job(job_id, status='packaging', task=task.task_id, checkpoint=str(checkpoint), base_version=base_version, federation_round=request.federation_round, epochs=request.epochs, **metrics)
            payload = pack_state_dict(state, metrics['classes'], task.architecture, base_version, task.task_id)
            persist_pending_update(task.task_id, request.federation_round, payload, {
                'job_id': job_id, 'base_global_model_version': base_version,
                'samples': metrics['samples'], 'architecture': task.architecture,
                'classes': metrics['classes'], 'checkpoint': str(checkpoint),
                'created_at': datetime.now(timezone.utc).isoformat(),
            })
            update_training_job(job_id, pending_update={'task': task.task_id, 'round': request.federation_round, 'status': 'pending'})
            
            if request.federation_api_url and request.federation_token:
                update_training_job(job_id, status='submitting')
                try:
                    submitted = submit_pending_update(task.task_id, request.federation_round, request.federation_api_url, request.federation_token)
                    completion = update_training_job(job_id, status='completed', submission={'status': 'submitted', 'round': request.federation_round, 'accepted_at': datetime.now(timezone.utc).isoformat(), 'bytes': submitted['bytes'], 'message': submitted.get('message', 'Model update accepted by central coordinator')})
                except (HTTPError, URLError, OSError, ValueError) as error:
                    completion = update_training_job(job_id, status='completed', submission={'status': 'pending_retry', 'round': request.federation_round, 'message': str(error)[:1000]})
            else:
                completion = update_training_job(job_id, status='completed', submission={'status': 'ready_to_submit', 'round': request.federation_round})
        else:
            completion = update_training_job(job_id, status='completed', task=task.task_id, checkpoint=str(checkpoint), base_version=base_version, epochs=request.epochs, **metrics)
        record_central_training_state(request, completion)
    except Exception as error:
        failed = update_training_job(job_id, status='failed', task=request.task, message=str(error), epochs=request.epochs)
        record_central_training_state(request, failed)
        return

@app.post('/train', status_code=202)
def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    if training_status.get('status') in {'queued', 'validating', 'training'}:
        raise HTTPException(409, 'A local training job is already running')
    try:
        task = get_task(request.task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    job_id = uuid4().hex
    if request.federation_round and request.base_global_model_version is None:
        raise HTTPException(400, 'Federated training requires the round base global model version')
    update_training_job(job_id, status='queued', task=task.task_id, created_at=datetime.now(timezone.utc).isoformat(), federation_round=request.federation_round, base_version=request.base_global_model_version)
    background_tasks.add_task(run_training, job_id, request)
    return {'job_id': job_id, 'status': 'queued', 'task': task.task_id, 'message': 'Local training started. Poll /training/status/{job_id} for progress.'}

@app.post('/training/reset')
def reset_training(task: str | None = None, clear_history: bool = True):
    """Reset training status when stuck or after an error."""
    global latest_training_job, training_status
    with training_jobs_lock:
        if clear_history:
            if task:
                to_delete = [jid for jid, j in training_jobs.items() if j.get('task') == task]
                for jid in to_delete:
                    del training_jobs[jid]
            else:
                training_jobs.clear()
            latest_training_job = None
        else:
            for job_id, job in training_jobs.items():
                if not task or job.get('task') == task:
                    if job.get('status') in {'queued', 'validating', 'training', 'packaging', 'submitting'}:
                        job['status'] = 'failed'
                        job['message'] = 'Local training was manually reset.'
                        job['updated_at'] = datetime.now(timezone.utc).isoformat()
        training_status = {'status': 'idle'}
        persist_training_jobs()
    return {'status': 'idle', 'message': 'Training status reset to idle'}

@app.post('/predict')
async def predict(file: UploadFile = File(...), task: str = 'PNEUMONIA', checkpoint_path: str | None = None):
    try:
        task_config = get_task(task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if not file.filename or Path(file.filename).suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(400, 'Only supported local image types may be predicted')
    if file.size is not None and file.size > MAX_PREDICTION_BYTES:
        raise HTTPException(413, 'The local image exceeds the size limit')
    try:
        with Image.open(file.file) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError('The local image exceeds the safety limit')
            opened.verify()
        file.file.seek(0)
        with Image.open(file.file) as opened:
            image = opened.convert('RGB')
        tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)
        checkpoint, saved, classes = select_inference_checkpoint(task_config.task_id, checkpoint_path)
        model = create_densenet(len(classes))
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())
        source = 'local-trained' if checkpoint.name == 'latest.pt' else 'global'
        return {'task': task_config.task_id, 'prediction': classes[index], 'confidence': float(probabilities[index]), 'model_version': saved.get('version', 'local'), 'model': checkpoint.name, 'model_source': source, 'timestamp': datetime.now(timezone.utc).isoformat()}
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        raise HTTPException(400, 'The local image could not be processed') from error

@app.post('/model/update')
def model_update(request: ModelUpdateRequest):
    try:
        task = get_task(request.task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    checkpoint = checkpoint_for_task(task.task_id, request.checkpoint_path, 'latest.pt')
    if not checkpoint.is_file():
        raise HTTPException(404, 'No local trained checkpoint is available')
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint_metadata(saved, task.task_id)
        state = saved['state_dict']
        metrics = saved.get('metrics', {})
        return {
            'parameters': {key: value.detach().cpu().reshape(-1).tolist() for key, value in state.items()},
            'parameter_shapes': {key: list(value.shape) for key, value in state.items()},
            'classes': saved['classes'], 'samples': metrics['samples'],
            'task': task.task_id, 'metrics': {'loss': metrics['loss'], 'accuracy': metrics['accuracy']}, 'architecture': saved.get('architecture', task.architecture), 'base_version': saved.get('base_version', 0),
        }
    except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise HTTPException(400, 'The local checkpoint could not be converted to a model update') from error

@app.post('/synchronize')
def synchronize(request: SynchronizeRequest):
    try:
        task = get_task(request.task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    checkpoint = checkpoint_for_task(task.task_id, request.checkpoint_path, 'global.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if task.class_labels and tuple(request.classes) != task.class_labels:
        raise HTTPException(400, 'Global model class labels do not match the selected medical task')
    if len(request.classes) < 2 or len(set(request.classes)) != len(request.classes):
        raise HTTPException(400, 'Global model class labels are invalid')
    try:
        state = {key: torch.tensor(value).reshape(request.parameter_shapes[key]) for key, value in request.state_dict.items()}
    except (KeyError, RuntimeError, ValueError) as error:
        raise HTTPException(400, 'Global model tensor shapes are invalid') from error
    torch.save({'task': task.task_id, 'state_dict': state, 'classes': request.classes, 'architecture': task.architecture, 'version': request.version, 'synchronized_at': datetime.now(timezone.utc).isoformat()}, checkpoint)
    return {'synchronized': True, 'task': task.task_id, 'checkpoint': str(checkpoint), 'version': request.version}

@app.post('/federated/submit')
def submit_update(request: SubmitUpdateRequest):
    try:
        task = get_task(request.task)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    checkpoint = checkpoint_for_task(task.task_id, request.checkpoint_path, 'latest.pt')
    if not checkpoint.is_file():
        raise HTTPException(404, 'No local trained checkpoint is available')
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint_metadata(saved, task.task_id)
        payload = pack_state_dict(saved['state_dict'], saved['classes'], saved.get('architecture', task.architecture), int(saved.get('base_version', 0)), task.task_id)
        persist_pending_update(task.task_id, request.round, payload, {
            'base_global_model_version': int(saved.get('base_version', 0)), 'samples': saved['metrics']['samples'],
            'architecture': saved.get('architecture', task.architecture), 'classes': saved['classes'], 'checkpoint': str(checkpoint),
        })
        result = submit_pending_update(task.task_id, request.round, request.api_url, request.token)
        if latest_training_job and training_jobs.get(latest_training_job, {}).get('task') == task.task_id:
            update_training_job(latest_training_job, submission={'status': 'submitted', 'round': request.round, 'accepted_at': datetime.now(timezone.utc).isoformat(), 'bytes': result.get('bytes')})
        return result
    except (OSError, RuntimeError, ValueError, HTTPError, URLError, KeyError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        raise HTTPException(502, 'Model update submission failed; the local AVM1 update is retained for retry: ' + detail[:1000]) from error


@app.get('/federated/pending')
def pending_updates(task: str = 'PNEUMONIA'):
    return local_state(task)['pending_updates']


@app.post('/federated/retry')
def retry_update(request: RetryUpdateRequest):
    try:
        task = get_task(request.task)
        result = submit_pending_update(task.task_id, request.round, request.api_url, request.token)
        if latest_training_job and training_jobs.get(latest_training_job, {}).get('task') == task.task_id:
            update_training_job(latest_training_job, submission={'status': 'submitted', 'round': request.round, 'accepted_at': datetime.now(timezone.utc).isoformat(), 'bytes': result.get('bytes')})
        return result
    except (OSError, RuntimeError, ValueError, HTTPError, URLError, KeyError) as error:
        detail = error.read().decode(errors='replace') if isinstance(error, HTTPError) else str(error)
        raise HTTPException(502, 'Model update retry failed; the local AVM1 update is retained: ' + detail[:1000]) from error


@app.post('/federated/bootstrap')
def bootstrap_global_model(request: BootstrapModelRequest):
    """Create the deterministic v0 AVM1 artifact without training on images."""
    try:
        task = get_task(request.task)
        classes = list(task.class_labels) if task.class_labels and tuple(c.upper() for c in request.classes) == tuple(c.upper() for c in task.class_labels) else request.classes
        if task.class_labels and tuple(c.upper() for c in classes) != tuple(c.upper() for c in task.class_labels):
            raise ValueError(f"Bootstrap class labels {classes} do not match canonical classes for {task.display_name}: {list(task.class_labels)}")
        if len(set(classes)) != len(classes) or any(not isinstance(label, str) or not label.strip() or len(label) > 256 for label in classes):
            raise ValueError('Bootstrap model class labels must be unique, non-empty strings')
        checkpoint = checkpoint_for_task(task.task_id, None, 'global.pt')
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            saved_classes = validate_checkpoint_metadata(saved, task.task_id)
            if int(saved.get('version', 0)) == 0 and saved_classes != classes:
                initialize_bootstrap_model(task.task_id, classes, checkpoint, overwrite=True)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
            elif int(saved.get('version', 0)) != 0:
                raise ValueError('The existing local global model cannot be used as this round\'s version-zero bootstrap')
            else:
                state = saved['state_dict']
        else:
            initialize_bootstrap_model(task.task_id, classes, checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        payload = pack_state_dict(state, classes, task.architecture, 0, task.task_id)
        return Response(content=payload, media_type='application/vnd.arogyavaani.model-v1', headers={'X-ArogyaVaani-Model-Version': '0', 'X-ArogyaVaani-Task': task.task_id})
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(400, 'Bootstrap model preparation failed: ' + str(error)) from error

@app.post('/federated/download-global')
def download_global(request: DownloadGlobalRequest):
    try:
        task = get_task(request.task)
        endpoint = request.api_url.rstrip('/') + '/api/federated/global-model-binary?task=' + task.task_id
        if request.expected_version is not None:
            endpoint += '&version=' + str(request.expected_version)
        try:
            response = urlopen(UrlRequest(endpoint, headers={'Authorization': 'Bearer ' + request.token, 'Accept': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
        except (URLError, OSError) as error:
            if 'localhost' in endpoint:
                fallback_endpoint = endpoint.replace('://localhost:', '://127.0.0.1:')
                response = urlopen(UrlRequest(fallback_endpoint, headers={'Authorization': 'Bearer ' + request.token, 'Accept': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
            else:
                raise
        payload = response.read()
        expected_checksum = response.headers.get('X-ArogyaVaani-Model-SHA256')
        if expected_checksum and not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_checksum):
            raise ValueError('The downloaded global model failed its integrity check')
        manifest = manifest_from_payload(payload)
        if manifest.get('task') != task.task_id or manifest.get('architecture') != task.architecture:
            raise ValueError('The synchronized model does not match the selected task')
        state_dict, classes = unpack_state_dict(payload)
        if task.class_labels and tuple(c.upper() for c in classes) != tuple(c.upper() for c in task.class_labels):
            raise ValueError(f'The synchronized model classes {classes} do not match the selected task {list(task.class_labels)}')
        if len(classes) < 2:
            raise ValueError('The synchronized model has invalid class labels')
        version = response.headers.get('X-ArogyaVaani-Model-Version', 'unknown')
        if response.headers.get('X-ArogyaVaani-Task') != task.task_id:
            raise ValueError('The synchronized model does not match the selected task')
        if request.expected_version is not None and version != str(request.expected_version):
            raise ValueError('The coordination API returned a different global model version')
        checkpoint = checkpoint_for_task(task.task_id, request.checkpoint_path, 'global.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'task': task.task_id, 'state_dict': state_dict, 'classes': classes, 'architecture': task.architecture, 'version': version, 'synchronized_at': datetime.now(timezone.utc).isoformat()}, checkpoint)
        return {'synchronized': True, 'task': task.task_id, 'checkpoint': str(checkpoint), 'version': version, 'bytes': len(payload)}
    except (OSError, RuntimeError, ValueError, HTTPError, URLError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        raise HTTPException(400, 'Global model download failed: ' + detail) from error

@app.post('/fedavg')
def aggregate(updates: list[Update]):
    if not updates:
        raise HTTPException(400, 'No model updates supplied')
    try:
        states = [({key: torch.tensor(value) for key, value in update.parameters.items()}, update.sample_count) for update in updates]
        result = weighted_fedavg(states)
        return {'parameters': {key: value.tolist() for key, value in result.items()}, 'sample_count': sum(update.sample_count for update in updates)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

if __name__ == '__main__':
    import sys
    from pathlib import Path
    import uvicorn
    ml_dir = str(Path(__file__).parent.resolve())
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    host = os.environ.get('HOST', '0.0.0.0' if os.environ.get('PORT') or os.environ.get('RENDER') else '127.0.0.1')
    port = int(os.environ.get('PORT', 8000))
    reload = os.environ.get('RELOAD', 'true').lower() in {'true', '1'} if not (os.environ.get('PORT') or os.environ.get('RENDER')) else False
    uvicorn.run('service:app', host=host, port=port, reload=reload, reload_dirs=[ml_dir] if reload else None)
