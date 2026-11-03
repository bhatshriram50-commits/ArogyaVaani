"""Central task identity/configuration for the hospital-local ML service."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class MedicalTask:
    task_id: str
    display_name: str
    organ: str
    modality: str
    architecture: str
    input_size: tuple[int, int]
    preprocessing: str = 'imagenet-rgb'
    # Empty labels intentionally defer to validated local dataset folders.
    class_labels: tuple[str, ...] = ()
    enabled: bool = True


TASKS = {
    'PNEUMONIA': MedicalTask('PNEUMONIA', 'Pneumonia', 'LUNG', 'CHEST_XRAY', 'densenet121', (224, 224), class_labels=('NORMAL', 'PNEUMONIA')),
    'RETINOPATHY': MedicalTask('RETINOPATHY', 'Retinopathy', 'EYE', 'FUNDUS_IMAGE', 'densenet121', (224, 224), class_labels=('Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe')),
    'BRAIN_TUMOR': MedicalTask('BRAIN_TUMOR', 'Brain Tumor', 'BRAIN', 'MRI', 'densenet121', (224, 224), class_labels=('glioma', 'meningioma', 'notumor', 'pituitary')),
}


def get_task(task_id: str) -> MedicalTask:
    task = TASKS.get(task_id)
    if not task or not task.enabled:
        raise ValueError('Medical task is invalid or disabled')
    return task


def task_checkpoint(task_id: str, name: str) -> str:
    get_task(task_id)
    root = Path(os.environ.get('LOCAL_MODEL_ROOT', 'local_models')).expanduser().resolve()
    return str(root / task_id.lower() / name)
