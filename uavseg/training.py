"""Bounded update control and streaming records, independent of model runtime."""

from dataclasses import asdict, dataclass
import math

from .common import AuditError, canonical


@dataclass(frozen=True)
class Budget:
    max_updates: int
    max_batches: int

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in (self.max_updates, self.max_batches)) or
                self.max_updates > self.max_batches):
            raise AuditError('更新数和读取批次数须为正整数，更新上限不能超过批次上限')


def _step_result(result):
    if not isinstance(result, dict) or type(result.get('updated')) is not bool:
        raise AuditError('单批更新结果缺少 updated 布尔值')
    valid = result.get('valid_pixels')
    if type(valid) is not int or valid < 0:
        raise AuditError('单批有效像素数无效')
    if result['updated']:
        loss = result.get('loss')
        if (valid == 0 or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0 or
                result.get('reason') != 'valid_pixels'):
            raise AuditError('成功更新须包含有效像素和有限非负损失')
    elif valid != 0 or result.get('loss') is not None or result.get('reason') != 'all_ignore':
        raise AuditError('跳过更新只允许全忽略批次，不能静默跳过其他错误')


def bounded_updates(batches, update, *, budget, emit):
    """Call update at most max_batches times, stopping after max_updates successes.

    emit receives immutable-by-convention JSON snapshots. An I/O failure stops
    execution immediately; no rollback or resumability is implied.
    """
    if not isinstance(budget, Budget) or not callable(update) or not callable(emit):
        raise AuditError('需要明确预算、更新函数和运行记录接收器')
    state = {'batches_consumed': 0, 'updates': 0, 'skipped_all_ignore': 0,
             'valid_pixels_updated': 0, 'last_loss': None}
    emit({'event': 'start', 'budget': asdict(budget), **state})
    try:
        iterator = iter(batches)
        reason = 'max_batches'
        while state['batches_consumed'] < budget.max_batches:
            # Check before next(): never read one extra official sample at the limit.
            if state['updates'] >= budget.max_updates:
                reason = 'max_updates'
                break
            try:
                batch = next(iterator)
            except StopIteration:
                reason = 'input_exhausted'
                break
            state['batches_consumed'] += 1
            emit({'event': 'batch_started', **state})
            result = update(batch)
            _step_result(result)
            if result['updated']:
                state['updates'] += 1
                state['valid_pixels_updated'] += result['valid_pixels']
                state['last_loss'] = float(result['loss'])
            else:
                state['skipped_all_ignore'] += 1
            emit({'event': 'batch_finished', **state, 'result': dict(result)})
        if state['updates'] >= budget.max_updates:
            reason = 'max_updates'
        summary = {'status': 'completed', 'stop_reason': reason, **state}
        emit({'event': 'end', **summary})
        return summary
    except BaseException as exc:
        # Preserve the original failure even if the record destination also fails.
        try:
            emit({'event': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                  'error_type': type(exc).__name__, **state})
        except BaseException:
            pass
        raise


class JsonlEvents:
    """Exclusive append-only local run journal; partial records survive failure."""
    def __init__(self, path, *, identity):
        from pathlib import Path
        from .common import check_output

        self.path = Path(path)
        local = Path(__file__).resolve().parent.parent / '.local'
        if not self.path.resolve().is_relative_to(local.resolve()):
            raise AuditError('运行记录须写入仓库 .local 目录')
        # Validate metadata before creating anything. Identity belongs to caller.
        self._header = canonical({'event': 'identity', 'identity': identity})
        check_output(self.path, [])
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('xb')
        try:
            self.stream.write(self._header)
            self.stream.flush()
        except BaseException:
            self.stream.close()
            raise
        return self

    def __call__(self, event):
        if self.stream is None or self.stream.closed:
            raise AuditError('运行记录器尚未打开或已经关闭')
        self.stream.write(canonical(event))
        self.stream.flush()

    def __exit__(self, *args):
        self.stream.close()
