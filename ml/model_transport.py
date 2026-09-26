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
TORCH_TO_WIRE = {
    torch.float32: 'f32', torch.float64: 'f64', torch.int64: 'i64',
    torch.int32: 'i32', torch.uint8: 'u8', torch.bool: 'bool',
}
WIRE_TO_TORCH = {value: key for key, value in TORCH_TO_WIRE.items()}


def pack_state_dict(state: OrderedDict, classes: list[str], architecture: str = 'densenet121', base_version: int = 0) -> bytes:
    if not architecture or len(architecture) > 128:
        raise ValueError('A valid model architecture identifier is required')
    if not isinstance(base_version, int) or base_version < 0:
        raise ValueError('base_version must be a non-negative integer')
    chunks: list[bytes] = []
    tensors = []
    offset = 0
    for name, tensor in state.items():
        if tensor.dtype not in TORCH_TO_WIRE:
            raise ValueError(f'Unsupported tensor dtype for {name}: {tensor.dtype}')
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        tensors.append({'name': name, 'shape': list(tensor.shape), 'dtype': TORCH_TO_WIRE[tensor.dtype], 'offset': offset, 'length': len(raw)})
        chunks.append(raw)
        offset += len(raw)
    manifest = json.dumps({'format': 'arogyavaani-state-v1', 'classes': classes, 'architecture': architecture, 'baseVersion': base_version, 'tensors': tensors}, separators=(',', ':')).encode()
    return MAGIC + struct.pack('<I', len(manifest)) + manifest + b''.join(chunks)


def unpack_state_dict(payload: bytes) -> tuple[OrderedDict, list[str]]:
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError('Invalid ArogyaVaani model payload')
    manifest_size = struct.unpack('<I', payload[4:8])[0]
    manifest_end = 8 + manifest_size
    manifest = json.loads(payload[8:manifest_end])
    if manifest.get('format') != 'arogyavaani-state-v1':
        raise ValueError('Unsupported model payload format')
    data = memoryview(payload)[manifest_end:]
    state = OrderedDict()
    for entry in manifest['tensors']:
        dtype = WIRE_TO_TORCH.get(entry['dtype'])
        if dtype is None or entry['offset'] < 0 or entry['length'] < 0 or entry['offset'] + entry['length'] > len(data):
            raise ValueError('Invalid tensor manifest')
        state[entry['name']] = torch.frombuffer(bytearray(data[entry['offset']:entry['offset'] + entry['length']]), dtype=dtype).reshape(entry['shape'])
    return state, manifest.get('classes', [])


def manifest_from_payload(payload: bytes) -> dict:
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError('Invalid ArogyaVaani model payload')
    size = struct.unpack('<I', payload[4:8])[0]
    return json.loads(payload[8:8 + size])
