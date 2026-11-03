from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
MAX_DATASET_FILES = 20_000
MAX_DATASET_UPLOAD_BYTES = 1_000_000_000
MAX_IMAGE_PIXELS = 40_000_000
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_densenet(num_classes: int) -> nn.Module:
    """Construct the shared v0 model without changing the caller's RNG state."""
    # Round one has no aggregated artifact.  A fixed seed makes its bootstrap
    # parameters identical at every selected hospital; later rounds load their
    # centrally aggregated checkpoint instead.
    with torch.random.fork_rng():
        torch.manual_seed(20261005)
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def validate_dataset(dataset_dir: str) -> dict:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('The local dataset directory does not exist')
    classes = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith('.'))
    if len(classes) < 2:
        raise ValueError('Dataset must contain at least two class folders')
    distribution, invalid, total_bytes, file_count = {}, [], 0, 0
    for class_dir in classes:
        files = [path for path in class_dir.rglob('*') if path.is_file()]
        file_count += len(files)
        if file_count > MAX_DATASET_FILES:
            raise ValueError(f'Dataset contains too many files (maximum {MAX_DATASET_FILES:,})')
        image_files = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(image_files) != len(files):
            raise ValueError(f'Class folder "{class_dir.name}" contains unsupported files')
        if not image_files:
            raise ValueError(f'Class folder "{class_dir.name}" has no supported images')
        distribution[class_dir.name] = len(image_files)
        for image_path in image_files:
            total_bytes += image_path.stat().st_size
            if total_bytes > MAX_DATASET_UPLOAD_BYTES:
                raise ValueError('Dataset exceeds the 1 GB local limit')
            try:
                from PIL import Image
                with Image.open(image_path) as image:
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError('image dimensions exceed the local safety limit')
                    image.verify()
            except (OSError, ValueError):
                invalid.append(str(image_path))
    if invalid:
        raise ValueError(f'Dataset has {len(invalid)} invalid image(s); correct them before training')
    return {'dataset_dir': str(root), 'classes': [item.name for item in classes], 'class_distribution': distribution, 'file_count': sum(distribution.values()), 'size_bytes': total_bytes, 'supported_types': sorted(IMAGE_SUFFIXES)}

def local_train(dataset_dir: str, num_classes: int, epochs: int = 10, batch_size: int = 16, initial_checkpoint: str | None = None, progress: Callable[[dict], None] | None = None) -> tuple[OrderedDict, dict]:
    validation = validate_dataset(dataset_dir)
    if len(validation['classes']) != num_classes:
        raise ValueError(f'num_classes ({num_classes}) does not match the {len(validation["classes"])} local class folders')
    dataset = datasets.ImageFolder(Path(dataset_dir), transform=IMAGENET_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = create_densenet(num_classes)
    if initial_checkpoint and Path(initial_checkpoint).is_file():
        saved = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
        ckpt_classes = saved.get('classes', [])
        if ckpt_classes and len(ckpt_classes) != num_classes:
            raise ValueError(f"Checkpoint classes count ({len(ckpt_classes)}) does not match dataset classes count ({num_classes})")
        state_dict = saved['state_dict'] if isinstance(saved, dict) and 'state_dict' in saved else saved
        classifier_weight = state_dict.get('classifier.weight') if isinstance(state_dict, dict) else None
        if classifier_weight is not None and classifier_weight.shape[0] != num_classes:
            raise ValueError(f"Checkpoint classifier output classes ({classifier_weight.shape[0]}) does not match model classes ({num_classes})")
        model.load_state_dict(state_dict)
    optimizer, criterion = torch.optim.Adam(model.parameters(), lr=1e-4), nn.CrossEntropyLoss()
    history = []
    total_batches = len(loader)
    model.train()
    if progress:
        progress({'status': 'training', 'epoch': 1, 'total_epochs': epochs, 'batch': 0, 'total_batches': total_batches, 'loss': 0.0, 'accuracy': 0.0})
    for epoch in range(1, epochs + 1):
        loss_sum, correct, count = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size_actual = labels.size(0)
            loss_sum += loss.item() * batch_size_actual
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size_actual
            if progress and (batch_idx == 1 or batch_idx % max(1, total_batches // 4) == 0 or batch_idx == total_batches):
                progress({
                    'status': 'training',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'batch': batch_idx,
                    'total_batches': total_batches,
                    'loss': loss_sum / count if count else 0.0,
                    'accuracy': correct / count if count else 0.0,
                })
        metric = {'epoch': epoch, 'total_epochs': epochs, 'batch': total_batches, 'total_batches': total_batches, 'loss': loss_sum / count if count else 0.0, 'accuracy': correct / count if count else 0.0}
        history.append(metric)
        if progress:
            progress({'status': 'training', **metric})
    # Report final per-class recall, not only aggregate training accuracy. A
    # balanced data set can still hide a model that predicts one class for
    # every image (the failure mode that must not be federated).
    model.eval()
    correct_by_class = [0] * num_classes
    total_by_class = [0] * num_classes
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            predictions = model(images).argmax(dim=1)
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                total_by_class[label] += 1
                correct_by_class[label] += int(label == prediction)
    class_accuracy = {
        dataset.classes[index]: {
            'correct': correct_by_class[index],
            'total': total_by_class[index],
            'accuracy': correct_by_class[index] / total_by_class[index] if total_by_class[index] else 0.0,
        }
        for index in range(num_classes)
    }
    evaluation_accuracy = sum(correct_by_class) / len(dataset) if len(dataset) else 0.0
    return model.state_dict(), {'samples': len(dataset), 'loss': history[-1]['loss'], 'accuracy': evaluation_accuracy, 'classes': dataset.classes, 'class_accuracy': class_accuracy, 'history': history, 'dataset': validation}

def weighted_fedavg(updates: Iterable[tuple[OrderedDict, int]]) -> OrderedDict:
    updates = list(updates)
    if not updates or any(samples <= 0 for _, samples in updates):
        raise ValueError('FedAvg requires at least one update with positive sample count')
    keys = list(updates[0][0].keys())
    if any(list(state.keys()) != keys for state, _ in updates):
        raise ValueError('FedAvg state dictionaries have different tensor keys')
    total_samples = sum(samples for _, samples in updates)
    result = OrderedDict()
    for key in keys:
        reference = updates[0][0][key]
        if any(state[key].shape != reference.shape for state, _ in updates):
            raise ValueError(f'FedAvg tensor shape mismatch for {key}')
        result[key] = sum(state[key].float() * samples for state, samples in updates) / total_samples
    return result

def save_global_model(state: OrderedDict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
