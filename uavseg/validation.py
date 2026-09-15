"""Deterministic validation traversal and internal checkpoint ranking."""

from dataclasses import dataclass
import re

from .common import AuditError
from .metrics import Confusion, INTERNAL_POLICY


def evaluate_samples(samples, predict, *, expected_ids, emit):
    """Evaluate each expected sample once, in order, using official label IDs."""
    if (not isinstance(expected_ids, tuple) or not expected_ids or
            any(not isinstance(value, str) or not value for value in expected_ids) or
            tuple(sorted(set(expected_ids))) != expected_ids or
            not callable(predict) or not callable(emit)):
        raise AuditError('验证需要非空、有序、无重复的预期ID及预测与记录函数')
    metric = Confusion(policy=INTERNAL_POLICY)
    emit({'event': 'validation_started', 'expected_samples': len(expected_ids),
          'metric_policy': INTERNAL_POLICY})
    iterator = iter(samples)
    try:
        for index, expected in enumerate(expected_ids, 1):
            try:
                sample = next(iterator)
            except StopIteration as exc:
                raise AuditError(f'验证样本提前结束，缺少 {expected}') from exc
            if not isinstance(sample, dict) or sample.get('id') != expected:
                raise AuditError(f'验证顺序或样本ID不匹配，预期 {expected}')
            if 'target' not in sample:
                raise AuditError(f'{expected}: 验证样本缺少标签')
            prediction = predict(sample)
            metric.update(sample['target'], prediction)
            emit({'event': 'validation_sample_finished', 'id': expected,
                  'samples_finished': index, 'valid_pixels': metric.valid_pixels,
                  'ignored_pixels': metric.ignored_pixels})
        sentinel = object()
        try:
            extra = next(iterator)
        except StopIteration:
            extra = sentinel
        if extra is not sentinel:
            name = extra.get('id') if isinstance(extra, dict) else '<invalid>'
            raise AuditError(f'验证输入包含预期范围外样本 {name}')
        report = metric.report()
        if report['status'] != 'valid':
            raise AuditError('验证集没有有效标签像素，不能产生模型分数')
        emit({'event': 'validation_finished', 'samples_finished': len(expected_ids),
              'valid_pixels': report['valid_pixels'], 'miou': report['miou'],
              'averaged_classes': report['averaged_classes']})
        return report
    except BaseException as exc:
        try:
            emit({'event': 'validation_interrupted' if isinstance(exc, KeyboardInterrupt)
                  else 'validation_failed', 'error_type': type(exc).__name__,
                  'samples_finished': metric.images})
        except BaseException:
            pass
        raise


@dataclass(frozen=True)
class SelectionContext:
    manifest_sha256: str
    split_sha256: str
    cadence_updates: int
    purpose: str = 'synthetic-selection-check'

    def __post_init__(self):
        for value in (self.manifest_sha256, self.split_sha256):
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                raise AuditError('模型选择上下文缺少数据或划分摘要')
        if type(self.cadence_updates) is not int or self.cadence_updates <= 0:
            raise AuditError('验证间隔须为正整数更新数')
        if self.purpose != 'synthetic-selection-check':
            raise AuditError('当前选择器只允许合成接口验证，不能启动实验选优')


class EarliestBest:
    """Retain the earliest strictly best valid internal score, without file I/O."""
    def __init__(self, context):
        if not isinstance(context, SelectionContext):
            raise AuditError('须提供明确的模型选择上下文')
        self.context = context
        self.records = []
        self.best = None

    def consider(self, *, updates, report, checkpoint_sha256):
        if (type(updates) is not int or updates <= 0 or
                updates % self.context.cadence_updates != 0 or
                (self.records and updates <= self.records[-1]['updates'])):
            raise AuditError('验证更新点须按固定间隔严格递增')
        if (not isinstance(checkpoint_sha256, str) or
                not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256)):
            raise AuditError('候选检查点摘要无效')
        if (not isinstance(report, dict) or report.get('policy') != INTERNAL_POLICY or
                report.get('official_equivalence_confirmed') is not False or
                report.get('status') != 'valid' or
                type(report.get('miou')) not in (int, float) or
                isinstance(report.get('miou'), bool) or
                not 0 <= report['miou'] <= 1 or
                type(report.get('averaged_classes')) is not int or
                not 1 <= report['averaged_classes'] <= 8):
            raise AuditError('候选验证报告不满足内部指标契约')
        record = {'updates': updates, 'miou': float(report['miou']),
                  'averaged_classes': report['averaged_classes'],
                  'checkpoint_sha256': checkpoint_sha256}
        self.records.append(record)
        selected = self.best is None or record['miou'] > self.best['miou']
        if selected:
            self.best = dict(record)
        return {'selected': selected, 'best': dict(self.best), 'evaluations': len(self.records),
                'purpose': self.context.purpose, 'official_model_selection': False}

    def report(self):
        return {'context': {'manifest_sha256': self.context.manifest_sha256,
                            'split_sha256': self.context.split_sha256,
                            'cadence_updates': self.context.cadence_updates,
                            'purpose': self.context.purpose},
                'records': [dict(row) for row in self.records],
                'best': None if self.best is None else dict(self.best),
                'official_model_selection': False}
