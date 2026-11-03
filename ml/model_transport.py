"""Binary transport for federated PyTorch state dictionaries.

Payloads contain a small JSON manifest followed by raw tensor bytes. They never
contain images, labels, paths, or patient metadata.
"""
from __future__ import annotations

import json
import struct
from collections import OrderedDict

import torch

MAGIC = b'AVM1'
MAX_MANIFEST_BYTES = 2_000_000
MAX_TENSORS = 20_000
MAX_TENSOR_ELEMENTS = 100_000_000
TORCH_TO_WIRE = {
    torch.float32: 'f32', torch.float64: 'f64', torch.int64: 'i64',
    torch.int32: 'i32', torch.uint8: 'u8', torch.bool: 'bool',
}
WIRE_TO_TORCH = {value: key for key, value in TORCH_TO_WIRE.items()}


def pack_state_dict(state: OrderedDict, classes: list[str], architecture: str = 'densenet121', base_version: int = 0, task: str = 'PNEUMONIA') -> bytes:
    if not architecture or len(architecture) > 128:
        raise ValueError('A valid model architecture identifier is required')
    if not isinstance(base_version, int) or base_version < 0:
        raise ValueError('base_version must be a non-negative integer')
    if not isinstance(classes, list) or not 2 <= len(classes) <= 1_000 or len(set(classes)) != len(classes) or not all(isinstance(label, str) and 0 < len(label) <= 256 for label in classes):
        raise ValueError('Valid, unique model class labels are required')
    if not state:
        raise ValueError('A model update must contain at least one tensor')
    chunks: list[bytes] = []
    tensors = []
    offset = 0
    names: set[str] = set()
    for name, tensor in state.items():
        if not isinstance(name, str) or not name or len(name) > 512 or name in names:
            raise ValueError('Model tensor names must be unique and valid')
        names.add(name)
        if tensor.dtype not in TORCH_TO_WIRE:
            raise ValueError(f'Unsupported tensor dtype for {name}: {tensor.dtype}')
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f'Model tensor {name} contains NaN or infinite values')
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        tensors.append({'name': name, 'shape': list(tensor.shape), 'dtype': TORCH_TO_WIRE[tensor.dtype], 'offset': offset, 'length': len(raw)})
        chunks.append(raw)
        offset += len(raw)
    if not task or len(task) > 128:
        raise ValueError('A valid medical task identifier is required')
    manifest = json.dumps({'format': 'arogyavaani-state-v1', 'task': task, 'classes': classes, 'architecture': architecture, 'baseVersion': base_version, 'tensors': tensors}, separators=(',', ':')).encode()
    return MAGIC + struct.pack('<I', len(manifest)) + manifest + b''.join(chunks)


def unpack_state_dict(payload: bytes) -> tuple[OrderedDict, list[str]]:
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError('Invalid ArogyaVaani model payload')
    manifest_size = struct.unpack('<I', payload[4:8])[0]
    manifest_end = 8 + manifest_size
    if manifest_size > MAX_MANIFEST_BYTES or manifest_end > len(payload):
        raise ValueError('Invalid model manifest size')
    try:
        manifest = json.loads(payload[8:manifest_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid model manifest') from error
    if manifest.get('format') != 'arogyavaani-state-v1' or not isinstance(manifest.get('tensors'), list) or not 1 <= len(manifest['tensors']) <= MAX_TENSORS:
        raise ValueError('Unsupported model payload format')
    classes = manifest.get('classes')
    if not isinstance(classes, list) or not 2 <= len(classes) <= 1_000 or len(set(classes)) != len(classes) or not all(isinstance(label, str) and 0 < len(label) <= 256 for label in classes):
        raise ValueError('Invalid model class labels')
    if not isinstance(manifest.get('architecture'), str) or not manifest['architecture'] or len(manifest['architecture']) > 128:
        raise ValueError('Invalid model architecture')
    if 'task' in manifest and (not isinstance(manifest['task'], str) or not manifest['task'] or len(manifest['task']) > 128):
        raise ValueError('Invalid model task identity')
    data = memoryview(payload)[manifest_end:]
    state = OrderedDict()
    names: set[str] = set()
    for entry in manifest['tensors']:
        if not isinstance(entry, dict) or not isinstance(entry.get('name'), str) or not entry['name'] or len(entry['name']) > 512 or entry['name'] in names or not isinstance(entry.get('shape'), list) or not all(isinstance(dimension, int) and 0 < dimension <= MAX_TENSOR_ELEMENTS for dimension in entry['shape']):
            raise ValueError('Invalid tensor manifest')
        names.add(entry['name'])
        elements = 1
        for dimension in entry['shape']:
            elements *= dimension
            if elements > MAX_TENSOR_ELEMENTS:
                raise ValueError('Tensor exceeds the safety limit')
        dtype = WIRE_TO_TORCH.get(entry.get('dtype'))
        if dtype is None or not isinstance(entry.get('offset'), int) or not isinstance(entry.get('length'), int) or entry['offset'] < 0 or entry['length'] < 0 or entry['offset'] + entry['length'] > len(data) or entry['length'] != elements * torch.empty((), dtype=dtype).element_size():
            raise ValueError('Invalid tensor manifest')
        tensor = torch.frombuffer(bytearray(data[entry['offset']:entry['offset'] + entry['length']]), dtype=dtype).reshape(entry['shape'])
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError('Model payload contains NaN or infinite values')
        state[entry['name']] = tensor
    return state, classes


def manifest_from_payload(payload: bytes) -> dict:
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError('Invalid ArogyaVaani model payload')
    size = struct.unpack('<I', payload[4:8])[0]
    if size > MAX_MANIFEST_BYTES or 8 + size > len(payload):
        raise ValueError('Invalid model manifest size')
    try:
        manifest = json.loads(payload[8:8 + size])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid model manifest') from error
    if not isinstance(manifest, dict) or manifest.get('format') != 'arogyavaani-state-v1':
        raise ValueError('Unsupported model payload format')
    return manifest
