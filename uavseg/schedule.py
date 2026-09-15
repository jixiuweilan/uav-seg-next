"""Pure deterministic sample scheduling; no image reads or framework state."""

import numpy as np

from .common import AuditError
from .data import sample_geometry


def training_schedule(ids, *, batch_size, seed, image_shape=(1024, 1024), crop_shape=(512, 512)):
    """Yield repeatable epochs forever; caller's bounded loop controls consumption."""
    if (not isinstance(ids, tuple) or not ids or tuple(sorted(set(ids))) != ids or
            any(not isinstance(value, str) or not value for value in ids)):
        raise AuditError('训练ID须为非空、有序、无重复元组')
    if type(batch_size) is not int or not 1 <= batch_size <= len(ids):
        raise AuditError('批量大小须为1至训练样本数的整数')
    if type(seed) is not int:
        raise AuditError('训练种子须为整数')
    rng = np.random.default_rng(seed)
    epoch = 0
    while True:
        order = rng.permutation(len(ids))
        for start in range(0, len(ids), batch_size):
            indices = order[start:start + batch_size]
            rows = []
            for index in indices:
                rows.append({'id': ids[int(index)],
                             'geometry': sample_geometry(image_shape, crop_shape, rng),
                             'epoch': epoch})
            yield tuple(rows)
        epoch += 1


def prepared_batches(samples, schedule):
    """Read one scheduled batch at a time and stack independent NumPy arrays."""
    if getattr(samples, 'partition', None) != 'train':
        raise AuditError('批次读取器只接受训练分区')
    for rows in schedule:
        if (not isinstance(rows, tuple) or not rows or
                any(not isinstance(row, dict) for row in rows)):
            raise AuditError('批次计划须为非空记录元组')
        ids = tuple(row.get('id') for row in rows)
        epochs = {row.get('epoch') for row in rows}
        if (any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids) or
                len(epochs) != 1 or any(type(value) is not int or value < 0 for value in epochs)):
            raise AuditError('单批ID须无重复且属于同一非负轮次')
        loaded = [samples.read(row['id'], geometry=row.get('geometry')) for row in rows]
        if any(not isinstance(sample, dict) or sample.get('id') != expected
               for sample, expected in zip(loaded, ids)):
            raise AuditError('读取结果与计划样本ID不一致')
        try:
            images = np.stack([sample['image'] for sample in loaded])
            targets = np.stack([sample['target'] for sample in loaded])
        except ValueError as exc:
            raise AuditError('单批图像或标签尺寸不一致') from exc
        if (images.dtype != np.float32 or targets.dtype != np.int64 or
                images.ndim != 4 or targets.shape != (len(rows), *images.shape[-2:])):
            raise AuditError('批次数组类型或尺寸不满足模型输入契约')
        yield {'ids': ids, 'epoch': next(iter(epochs)),
               'images': np.ascontiguousarray(images),
               'targets': np.ascontiguousarray(targets)}
