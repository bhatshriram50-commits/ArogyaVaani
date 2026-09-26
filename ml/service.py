from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from federated import IMAGE_SUFFIXES, IMAGENET_TRANSFORM, MAX_DATASET_FILES, MAX_DATASET_UPLOAD_BYTES, create_densenet, local_train, validate_dataset, weighted_fedavg
from model_transport import pack_state_dict, unpack_state_dict

app = FastAPI(title='ArogyaVaani Local Training Service')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])
training_status: dict = {'status': 'idle'}
DATASET_UPLOAD_ROOT = Path('local_datasets/uploads').resolve()

class Update(BaseModel):
    sample_count: int = Field(gt=0)
    parameters: dict[str, list[float]]

class TrainingRequest(BaseModel):
    dataset_dir: str
    num_classes: int = Field(gt=1)
    epochs: int = Field(default=1, ge=1, le=100)
    checkpoint_path: str = 'local_models/latest.pt'
    global_checkpoint_path: str | None = 'local_models/global.pt'

class SynchronizeRequest(BaseModel):
    checkpoint_path: str = 'local_models/global.pt'
    state_dict: dict[str, list[float]]
    parameter_shapes: dict[str, list[int]] = {}
    classes: list[str] = []
    version: int = Field(ge=0)

class SubmitUpdateRequest(BaseModel):
    api_url: str
    token: str = Field(min_length=1)
    round: int = Field(gt=0)
    checkpoint_path: str = 'local_models/latest.pt'

class DownloadGlobalRequest(BaseModel):
    api_url: str
    token: str = Field(min_length=1)
    checkpoint_path: str = 'local_models/global.pt'
    expected_version: int | None = Field(default=None, ge=0)

class ModelUpdateRequest(BaseModel):
    checkpoint_path: str = 'local_models/latest.pt'

@app.get('/health')
def health():
    return {'status': 'operational', 'privacy': 'images stay local'}

@app.get('/dataset/validate')
def dataset_validation(dataset_dir: str):
    try:
        return validate_dataset(dataset_dir)
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
async def upload_dataset(files: list[UploadFile] = File(...)):
    """Receive a browser-selected dataset folder into hospital-local storage only."""
    if not files:
        raise HTTPException(400, 'Select a dataset folder')
    if len(files) > MAX_DATASET_FILES:
        raise HTTPException(400, f'Dataset contains too many files (maximum {MAX_DATASET_FILES:,})')
    upload_dir = DATASET_UPLOAD_ROOT / uuid4().hex
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
        return {'dataset_dir': str(dataset_dir), 'dataset_name': dataset_name, 'validation': validation, 'storage': 'local-only'}
    except (OSError, ValueError) as error:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(400, str(error) if str(error) else 'Dataset upload could not be processed') from error

@app.get('/training/status')
def get_training_status():
    return training_status

def run_training(request: TrainingRequest):
    global training_status
    try:
        training_status = {'status': 'validating'}
        def update_progress(progress: dict):
            global training_status
            training_status = progress
        base_version = 0
        if request.global_checkpoint_path and Path(request.global_checkpoint_path).is_file():
            global_checkpoint = torch.load(request.global_checkpoint_path, map_location='cpu', weights_only=True)
            base_version = int(global_checkpoint.get('version', 0))
        state, metrics = local_train(request.dataset_dir, request.num_classes, request.epochs, initial_checkpoint=request.global_checkpoint_path, progress=update_progress)
        checkpoint = Path(request.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': state, 'classes': metrics['classes'], 'metrics': metrics, 'architecture': 'densenet121', 'base_version': base_version, 'created_at': datetime.now(timezone.utc).isoformat()}, checkpoint)
        training_status = {'status': 'completed', 'checkpoint': str(checkpoint), 'base_version': base_version, **metrics}
    except (OSError, RuntimeError, ValueError) as error:
        training_status = {'status': 'failed', 'message': str(error)}
        return

@app.post('/train', status_code=202)
def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    if training_status.get('status') in {'validating', 'training'}:
        raise HTTPException(409, 'A local training job is already running')
    background_tasks.add_task(run_training, request)
    return {'status': 'queued', 'message': 'Local training started. Poll /training/status for progress.'}

@app.post('/predict')
async def predict(file: UploadFile = File(...), checkpoint_path: str = 'local_models/global.pt'):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        checkpoint = Path('local_models/latest.pt')
    if not checkpoint.is_file():
        raise HTTPException(404, 'No synchronized global or local model checkpoint is available')
    try:
        image = Image.open(file.file).convert('RGB')
        tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        classes = saved['classes']
        model = create_densenet(len(classes))
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())
        return {'prediction': classes[index], 'confidence': float(probabilities[index]), 'model_version': saved.get('version', 'local'), 'model': checkpoint.name, 'timestamp': datetime.now(timezone.utc).isoformat()}
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        raise HTTPException(400, 'The local image could not be processed') from error

@app.post('/model/update')
def model_update(request: ModelUpdateRequest):
    checkpoint = Path(request.checkpoint_path)
    if not checkpoint.is_file():
        raise HTTPException(404, 'No local trained checkpoint is available')
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = saved['state_dict']
        metrics = saved.get('metrics', {})
        return {
            'parameters': {key: value.detach().cpu().reshape(-1).tolist() for key, value in state.items()},
            'parameter_shapes': {key: list(value.shape) for key, value in state.items()},
            'classes': saved['classes'], 'samples': metrics['samples'],
            'metrics': {'loss': metrics['loss'], 'accuracy': metrics['accuracy']}, 'architecture': saved.get('architecture', 'densenet121'), 'base_version': saved.get('base_version', 0),
        }
    except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise HTTPException(400, 'The local checkpoint could not be converted to a model update') from error

@app.post('/synchronize')
def synchronize(request: SynchronizeRequest):
    checkpoint = Path(request.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = {key: torch.tensor(value).reshape(request.parameter_shapes[key]) for key, value in request.state_dict.items()}
    except (KeyError, RuntimeError, ValueError) as error:
        raise HTTPException(400, 'Global model tensor shapes are invalid') from error
    torch.save({'state_dict': state, 'classes': request.classes, 'version': request.version, 'synchronized_at': datetime.now(timezone.utc).isoformat()}, checkpoint)
    return {'synchronized': True, 'checkpoint': str(checkpoint), 'version': request.version}

@app.post('/federated/submit')
def submit_update(request: SubmitUpdateRequest):
    checkpoint = Path(request.checkpoint_path)
    if not checkpoint.is_file():
        raise HTTPException(404, 'No local trained checkpoint is available')
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        payload = pack_state_dict(saved['state_dict'], saved['classes'], saved.get('architecture', 'densenet121'), int(saved.get('base_version', 0)))
        # The coordination API derives the hospital identity from the signed JWT;
        # a local client must never supply a separate identity claim.
        endpoint = request.api_url.rstrip('/') + '/api/federated/rounds/' + str(request.round) + '/updates?samples=' + str(saved['metrics']['samples'])
        response = urlopen(Request(endpoint, data=payload, method='POST', headers={'Authorization': 'Bearer ' + request.token, 'Content-Type': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
        return {'submitted': True, **__import__('json').loads(response.read()), 'bytes': len(payload)}
    except (OSError, RuntimeError, ValueError, HTTPError, URLError, KeyError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        raise HTTPException(400, 'Model update submission failed: ' + detail) from error

@app.post('/federated/download-global')
def download_global(request: DownloadGlobalRequest):
    try:
        endpoint = request.api_url.rstrip('/') + '/api/federated/global-model-binary'
        if request.expected_version is not None:
            endpoint += '?version=' + str(request.expected_version)
        response = urlopen(Request(endpoint, headers={'Authorization': 'Bearer ' + request.token, 'Accept': 'application/vnd.arogyavaani.model-v1'}), timeout=60)
        payload = response.read()
        state_dict, classes = unpack_state_dict(payload)
        version = response.headers.get('X-ArogyaVaani-Model-Version', 'unknown')
        if request.expected_version is not None and version != str(request.expected_version):
            raise ValueError('The coordination API returned a different global model version')
        checkpoint = Path(request.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': state_dict, 'classes': classes, 'version': version, 'synchronized_at': datetime.now(timezone.utc).isoformat()}, checkpoint)
        return {'synchronized': True, 'checkpoint': str(checkpoint), 'version': version, 'bytes': len(payload)}
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
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
