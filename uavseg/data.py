"""NumPy-only baseline geometry and label mapping; no image I/O or training."""

from dataclasses import asdict, dataclass

import numpy as np

from .common import AuditError

IGNORE_TARGET = -100


@dataclass(frozen=True)
class Geometry:
    top: int
    left: int
    height: int
    width: int
    horizontal_flip: bool = False
    vertical_flip: bool = False
    quarter_turns: int = 0


def _mask(mask):
    value = np.asarray(mask)
    if value.dtype != np.uint8 or value.ndim != 2 or not value.size or np.any(value > 8):
        raise AuditError('标签须为非空二维 uint8，取值0至8')
    return value


def encode_targets(mask):
    value = _mask(mask)
    targets = value.astype(np.int64) - 1
    targets[value == 0] = IGNORE_TARGET
    return targets


def decode_predictions(indices):
    value = np.asarray(indices)
    if (value.ndim != 2 or not value.size or value.dtype.kind not in 'iu' or
            np.any(value > 7) or np.any(value < 0)):
        raise AuditError('预测类别索引须为非空二维整数，取值0至7')
    return (value + 1).astype(np.uint8)


def _shape(shape):
    if (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)):
        raise AuditError('尺寸须为两个正整数')
    return shape


def sample_geometry(image_shape, crop_shape, rng):
    """Consume an explicit local Generator once for the shared image/mask draw."""
    height, width = _shape(image_shape)
    crop_h, crop_w = _shape(crop_shape)
    if crop_h > height or crop_w > width:
        raise AuditError('裁剪尺寸超出原图')
    if not isinstance(rng, np.random.Generator):
        raise AuditError('须提供独立的 NumPy Generator')
    return Geometry(int(rng.integers(height - crop_h + 1)), int(rng.integers(width - crop_w + 1)),
                    crop_h, crop_w, bool(rng.integers(2)), bool(rng.integers(2)),
                    int(rng.integers(4)))


def transform_pair(image, mask, geometry):
    image, mask = np.asarray(image), _mask(mask)
    if image.dtype != np.uint8 or image.shape != (*mask.shape, 3):
        raise AuditError('图像须为 uint8 RGB，且与标签尺寸一致')
    if not isinstance(geometry, Geometry):
        raise AuditError('须提供共享的几何变换记录')
    g = geometry
    if (any(type(v) is not int for v in (g.top, g.left, g.height, g.width, g.quarter_turns)) or
            type(g.horizontal_flip) is not bool or type(g.vertical_flip) is not bool or
            g.top < 0 or g.left < 0 or g.height <= 0 or g.width <= 0 or
            g.top + g.height > mask.shape[0] or g.left + g.width > mask.shape[1] or
            g.quarter_turns not in range(4)):
        raise AuditError('几何变换越界或参数无效')

    def apply(value):
        result = value[g.top:g.top + g.height, g.left:g.left + g.width]
        if g.horizontal_flip:
            result = np.flip(result, axis=1)
        if g.vertical_flip:
            result = np.flip(result, axis=0)
        result = np.rot90(result, g.quarter_turns, axes=(0, 1))
        return result.copy(order='C')

    return apply(image), apply(mask)


def prepare_pair(image, mask, geometry=None):
    """Return CHW float32 RGB and int64 targets, retaining all-ignore examples."""
    mask = _mask(mask)
    g = geometry if geometry is not None else Geometry(0, 0, *mask.shape)
    pixels, labels = transform_pair(image, mask, g)
    normalized = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)
    normalized /= np.float32(255)
    return {'image': normalized, 'target': encode_targets(labels),
            'valid_pixels': int(np.count_nonzero(labels)), 'geometry': asdict(g)}
