"""Budget termination and failure records use synthetic callbacks, no torch."""

import json
from pathlib import Path
import tempfile
import unittest

from uavseg.common import AuditError
from uavseg.training import Budget, JsonlEvents, bounded_updates


def result(updated):
    return {'updated': updated, 'reason': 'valid_pixels' if updated else 'all_ignore',
            'valid_pixels': 10 if updated else 0, 'loss': 1.25 if updated else None}


class TrainingControlTests(unittest.TestCase):
    def test_update_limit_never_consumes_an_extra_batch(self):
        events = []
        def batches():
            yield False
            yield True
            yield True
            self.fail('read beyond update limit')
        summary = bounded_updates(batches(), result, budget=Budget(2, 10), emit=events.append)
        self.assertEqual(summary['stop_reason'], 'max_updates')
        self.assertEqual(summary['updates'], 2)
        self.assertEqual(summary['batches_consumed'], 3)
        self.assertEqual(summary['skipped_all_ignore'], 1)
        self.assertEqual(summary['valid_pixels_updated'], 20)
        self.assertEqual(events[0]['updates'], 0)  # Earlier snapshots remain unchanged.
        self.assertEqual(events[-1]['event'], 'end')

    def test_infinite_all_ignore_stops_at_batch_limit(self):
        import itertools
        summary = bounded_updates(itertools.repeat(False), result, budget=Budget(2, 3), emit=lambda e: None)
        self.assertEqual(summary['updates'], 0)
        self.assertEqual(summary['skipped_all_ignore'], 3)
        self.assertEqual(summary['stop_reason'], 'max_batches')
        self.assertIsNone(summary['last_loss'])

    def test_exhausted_and_empty_input_are_explicit(self):
        for values in ([], [True]):
            summary = bounded_updates(values, result, budget=Budget(2, 3), emit=lambda e: None)
            self.assertEqual(summary['stop_reason'], 'input_exhausted')
            self.assertEqual(summary['batches_consumed'], len(values))

    def test_failures_are_recorded_and_reraised_without_retry(self):
        for error in (RuntimeError('synthetic'), KeyboardInterrupt()):
            events, calls = [], []
            def failing_step(value):
                calls.append(value)
                raise error
            with self.assertRaises(type(error)):
                bounded_updates([1, 2], failing_step, budget=Budget(2, 3), emit=events.append)
            self.assertEqual(calls, [1])
            self.assertEqual(events[-1]['batches_consumed'], 1)
            self.assertEqual(events[-1]['updates'], 0)
            self.assertIn(events[-1]['event'], ('failed', 'interrupted'))

    def test_invalid_budget_and_step_contract_fail(self):
        for values in ((0, 1), (2, 1), (True, 2), (1, 2.0)):
            with self.assertRaises(AuditError):
                Budget(*values)
        for value in ({}, {**result(True), 'loss': float('nan')},
                      {**result(False), 'reason': 'corrupt_file'}, {**result(True), 'valid_pixels': 0}):
            with self.assertRaises(AuditError):
                bounded_updates([1], lambda batch: value, budget=Budget(1, 1), emit=lambda e: None)

    def test_journal_preserves_partial_records_and_refuses_overwrite(self):
        local = Path(__file__).resolve().parent.parent / '.local'
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            path = Path(temporary) / 'events.jsonl'
            with self.assertRaises(KeyboardInterrupt):
                with JsonlEvents(path, identity={'purpose': 'synthetic'}) as emit:
                    def interrupted(batch):
                        raise KeyboardInterrupt()
                    bounded_updates([1], interrupted, budget=Budget(1, 1), emit=emit)
            records = [json.loads(line) for line in path.read_bytes().splitlines()]
            self.assertEqual([r['event'] for r in records], ['identity', 'start', 'batch_started', 'interrupted'])
            before = path.read_bytes()
            with self.assertRaises(AuditError):
                JsonlEvents(path, identity={})
            self.assertEqual(path.read_bytes(), before)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(AuditError):
                JsonlEvents(Path(temporary) / 'events.jsonl', identity={})

    def test_recording_failure_stops_before_another_update(self):
        calls = []
        def emit(event):
            if event['event'] == 'batch_finished':
                raise OSError('disk full')
        def step(value):
            calls.append(value)
            return result(True)
        with self.assertRaisesRegex(OSError, 'disk full'):
            bounded_updates([1, 2], step, budget=Budget(2, 2), emit=emit)
        self.assertEqual(calls, [1])
