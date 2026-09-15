"""Training schedules are deterministic and keep each epoch complete."""

import unittest

from uavseg.common import AuditError
from uavseg.schedule import prepared_batches, training_schedule


class ScheduleTests(unittest.TestCase):
    def test_seed_replay_batches_and_epoch_coverage(self):
        ids = tuple('abcde')
        first, second = training_schedule(ids, batch_size=2, seed=7), training_schedule(ids, batch_size=2, seed=7)
        left, right = [next(first) for _ in range(6)], [next(second) for _ in range(6)]
        self.assertEqual(left, right)
        for epoch, rows in enumerate((left[:3], left[3:])):
            flat = [item for batch in rows for item in batch]
            self.assertEqual({item['id'] for item in flat}, set(ids))
            self.assertEqual({item['epoch'] for item in flat}, {epoch})
            self.assertEqual([len(batch) for batch in rows], [2, 2, 1])
            self.assertTrue(all(item['geometry'].height == 512 for item in flat))

    def test_seed_changes_order_or_geometry(self):
        ids = tuple('abcdef')
        self.assertNotEqual(next(training_schedule(ids, batch_size=3, seed=0)),
                            next(training_schedule(ids, batch_size=3, seed=1)))

    def test_bad_contracts_fail_before_iteration(self):
        cases = ((['a'], 1, 0), (('b', 'a'), 1, 0), (('a', 'a'), 1, 0),
                 (('a',), 0, 0), (('a',), 2, 0), (('a',), 1, True))
        for ids, size, seed in cases:
            iterator = training_schedule(ids, batch_size=size, seed=seed)
            with self.assertRaises(AuditError):
                next(iterator)

    def test_prepared_batches_preserve_order_and_stack_types(self):
        import numpy as np
        from uavseg.data import Geometry
        class Samples:
            partition = 'train'
            def __init__(self):
                self.calls = []
            def read(self, sample_id, *, geometry):
                self.calls.append((sample_id, geometry))
                return {'id': sample_id, 'image': np.zeros((3, 4, 5), np.float32),
                        'target': np.zeros((4, 5), np.int64)}
        samples = Samples()
        geometry = Geometry(0, 0, 4, 5)
        batch = next(prepared_batches(samples, [({'id': 'b', 'geometry': geometry, 'epoch': 2},
                                                  {'id': 'a', 'geometry': geometry, 'epoch': 2})]))
        self.assertEqual(batch['ids'], ('b', 'a'))
        self.assertEqual(batch['images'].shape, (2, 3, 4, 5))
        self.assertEqual(batch['targets'].shape, (2, 4, 5))
        self.assertEqual(samples.calls, [('b', geometry), ('a', geometry)])

    def test_prepared_batch_rejects_bad_plan_or_result(self):
        import numpy as np
        from uavseg.data import Geometry
        class Samples:
            partition = 'train'
            def read(self, sample_id, *, geometry):
                dtype = np.float64 if sample_id == 'bad' else np.float32
                return {'image': np.zeros((3, 2, 2), dtype), 'target': np.zeros((2, 2), np.int64)}
        geometry = Geometry(0, 0, 2, 2)
        for rows in ([], [{'id': 'a'}], ({'id': 'a', 'geometry': geometry, 'epoch': 0},
                                         {'id': 'a', 'geometry': geometry, 'epoch': 0}),
                     ({'id': 'a', 'geometry': geometry, 'epoch': 0},
                      {'id': 'b', 'geometry': geometry, 'epoch': 1}),
                     ({'id': 'bad', 'geometry': geometry, 'epoch': 0},)):
            with self.assertRaises(AuditError):
                next(prepared_batches(Samples(), [rows]))
        samples = Samples()
        samples.partition = 'val'
        with self.assertRaises(AuditError):
            next(prepared_batches(samples, []))
        class Wrong(Samples):
            def read(self, sample_id, *, geometry):
                return {**super().read(sample_id, geometry=geometry), 'id': 'other'}
        with self.assertRaisesRegex(AuditError, '样本ID'):
            next(prepared_batches(Wrong(), [({'id': 'a', 'geometry': geometry, 'epoch': 0},)]))
