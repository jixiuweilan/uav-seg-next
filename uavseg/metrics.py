"""Explicit internal metric policy, not an organizer-certified evaluator."""

import numpy as np

from .common import AuditError

INTERNAL_POLICY = 'internal-global-ignore0-v1'


class Confusion:
    def __init__(self, *, policy):
        if policy != INTERNAL_POLICY:
            raise AuditError('须显式选择内部指标口径；尚未确认官方边界规则')
        self.policy = policy
        self._counts = np.zeros((9, 9), dtype=np.int64)
        self.images = 0
        self.valid_pixels = 0
        self.ignored_pixels = 0

    def update(self, target, prediction):
        target, prediction = np.asarray(target), np.asarray(prediction)
        if target.shape != prediction.shape or target.ndim != 2 or not target.size:
            raise AuditError('指标输入须为形状相同的非空二维标签')
        for value in (target, prediction):
            if value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value > 8):
                raise AuditError('指标使用官方整数标签0至8，不接受模型索引或浮点数')
        valid = target != 0
        n = int(np.count_nonzero(valid))
        if self.valid_pixels + n > np.iinfo(np.int64).max:
            raise AuditError('累计有效像素数超出 int64')
        codes = target[valid].astype(np.int64) * 9 + prediction[valid].astype(np.int64)
        delta = np.bincount(codes, minlength=81).reshape(9, 9)
        self._counts += delta
        self.images += 1
        self.valid_pixels += n
        self.ignored_pixels += int(target.size) - n

    def report(self):
        # Python integers also prevent overflow in row + column before subtraction.
        counts = self._counts.tolist()
        classes = []
        for k in range(1, 9):
            support = sum(counts[k])
            predicted = sum(row[k] for row in counts)
            intersection = counts[k][k]
            union = support + predicted - intersection
            classes.append({'id': k, 'target_pixels': support, 'predicted_pixels': predicted,
                            'intersection': intersection, 'union': union,
                            'iou': intersection / union if union else None})
        ious = [row['iou'] for row in classes if row['iou'] is not None]
        return {'policy': self.policy, 'official_equivalence_confirmed': False,
                'status': 'valid' if self.valid_pixels else 'invalid_no_valid_pixels',
                'images': self.images, 'valid_pixels': self.valid_pixels,
                'ignored_pixels': self.ignored_pixels, 'confusion': counts,
                'classes': classes, 'averaged_classes': len(ious),
                'miou': sum(ious) / len(ious) if ious else None}
