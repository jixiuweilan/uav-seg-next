"""Execution-side baseline primitives; no formal training entrypoint."""

from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from .common import AuditError, write_binary
from .data import IGNORE_TARGET, decode_predictions
from .metrics import INTERNAL_POLICY
from .model import CompactUNet, MODEL_ID
from .submission import file_sha256
from .training import bounded_updates


def _images(images):
    if (images.dtype != torch.float32 or images.ndim != 4 or images.shape[0] < 1 or
            images.shape[1] != 3 or min(images.shape[-2:]) < 16 or
            not torch.isfinite(images).all().item() or
            images.min().item() < 0 or images.max().item() > 1):
        raise AuditError('输入须为有限的 float32 RGB 批次，归一化到0至1')


def update_batch(model, optimizer, images, targets):
    """One explicit update, for synthetic validation now; all-ignore has no step."""
    _images(images)
    if (targets.dtype != torch.int64 or targets.device != images.device or
            targets.shape != (images.shape[0], *images.shape[-2:]) or
            not (((targets >= 0) & (targets < 8)) | (targets == IGNORE_TARGET)).all().item()):
        raise AuditError('目标须为同设备的 int64 标签，取值0至7或-100')
    valid = int((targets != IGNORE_TARGET).sum().item())
    optimizer.zero_grad(set_to_none=True)
    if not valid:
        return {'updated': False, 'reason': 'all_ignore', 'valid_pixels': 0, 'loss': None}
    model.train()
    logits = model(images)
    loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_TARGET)
    if not torch.isfinite(loss).item():
        raise AuditError('损失非有限，未更新参数')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise AuditError('梯度非有限，未更新参数')
    optimizer.step()
    return {'updated': True, 'reason': 'valid_pixels', 'valid_pixels': valid,
            'loss': float(loss.detach().cpu())}


def update_batches(model, optimizer, batches, *, budget, emit):
    """Consume explicit (images, targets) batches with both limits and event records."""
    return bounded_updates(batches, lambda pair: update_batch(model, optimizer, *pair),
                           budget=budget, emit=emit)


def predict_image(model, image, *, device='cpu'):
    """Full-resolution single-image prediction; never accepts or opens labels."""
    value = np.asarray(image)
    if value.dtype != np.float32 or value.shape != (3, 1024, 1024):
        raise AuditError('预测输入须为未经缩放的3×1024×1024 float32 图像')
    tensor = torch.from_numpy(np.array(value, copy=True, order='C')).unsqueeze(0).to(device)
    _images(tensor)
    previous = model.training
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(tensor)
            if logits.shape != (1, 8, 1024, 1024) or not torch.isfinite(logits).all().item():
                raise AuditError('预测输出尺寸或数值无效')
            return decode_predictions(logits.argmax(dim=1)[0].cpu().numpy())
    finally:
        model.train(previous)


def _provenance(value):
    required = {'manifest_sha256', 'split_sha256', 'source_sha256', 'metric_policy', 'seed', 'purpose'}
    if not isinstance(value, dict) or set(value) != required:
        raise AuditError('权重须绑定数据、划分、代码、指标口径、种子及用途')
    for key in ('manifest_sha256', 'split_sha256', 'source_sha256'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise AuditError('权重来源摘要无效')
    if (value['metric_policy'] != INTERNAL_POLICY or type(value['seed']) is not int or
            value['purpose'] != 'synthetic-runtime-check'):
        raise AuditError('当前仅接收合成运行验证权重，不作为正式训练或提交权重')
    return dict(value)


def save_checkpoint(model, output, *, provenance, protected):
    if type(model) is not CompactUNet:
        raise AuditError('仅支持已声明的单一基线模型')
    provenance = _provenance(provenance)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if any(not value.is_floating_point() or not torch.isfinite(value).all().item()
           for value in state.values()):
        raise AuditError('权重张量无效')
    payload = {'schema_version': 1, 'model_id': MODEL_ID, 'provenance': provenance,
               'torch_version': str(torch.__version__), 'state_dict': state}
    # A dot-prefixed suffixless path gives PyTorch's filename writer an empty
    # archive basename. Its stream writer does not derive names from the path.
    write_binary(output, lambda stream: torch.save(payload, stream), protected)
    return {'checkpoint_sha256': file_sha256(Path(output)), 'provenance': provenance}


def load_checkpoint(path, *, expected_sha256, provenance):
    provenance = _provenance(provenance)
    # Hash and deserialize the same open file; load only our known tensor schema.
    import hashlib
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected_sha256:
            raise AuditError('权重文件摘要不匹配')
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1 or
            payload.get('model_id') != MODEL_ID or payload.get('provenance') != provenance or
            not isinstance(payload.get('state_dict'), dict)):
        raise AuditError('权重结构或运行来源不匹配')
    # Constructing a loader must not consume the caller's global CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        model = CompactUNet()
    expected = model.state_dict()
    state = payload['state_dict']
    if set(state) != set(expected):
        raise AuditError('权重参数名不匹配')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or
                value.dtype != expected[key].dtype or not torch.isfinite(value).all().item()):
            raise AuditError('权重张量尺寸、类型或数值不匹配')
    model.load_state_dict(state, strict=True)
    return model.eval()
