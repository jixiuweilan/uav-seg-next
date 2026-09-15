"""Optional bounded-loop integration checks for a future execution-side batch."""

import unittest

import torch

from uavseg.common import AuditError
from uavseg.model import CompactUNet
from uavseg.runtime import tensor_batches, update_batches
from uavseg.training import Budget


class BoundedRuntimeTests(unittest.TestCase):
    def test_numpy_batches_copy_to_explicit_device(self):
        import numpy as np
        images = np.zeros((1, 3, 16, 17), np.float32)
        targets = np.zeros((1, 16, 17), np.int64)
        value = next(tensor_batches([{'ids': ('a',), 'epoch': 0,
                                      'images': images, 'targets': targets}], device='cpu'))
        images[:] = 1
        targets[:] = 2
        self.assertTrue(torch.all(value[0] == 0).item())
        self.assertTrue(torch.all(value[1] == 0).item())
        for bad in ({'ids': (), 'epoch': 0, 'images': images, 'targets': targets},
                    {'ids': ('a',), 'epoch': -1, 'images': images, 'targets': targets},
                    {'ids': ('a',), 'epoch': 0, 'images': images.astype(np.float64), 'targets': targets}):
            with self.assertRaises(AuditError):
                next(tensor_batches([bad], device='cpu'))

    def test_skip_then_update_stops_without_reading_next_batch(self):
        torch.manual_seed(0)
        model = CompactUNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        image = torch.zeros((1, 3, 32, 32))
        targets = torch.zeros((1, 32, 32), dtype=torch.int64)
        def batches():
            yield image, torch.full_like(targets, -100)
            yield image, targets
            self.fail('batch read after successful-update limit')
        events = []
        summary = update_batches(model, optimizer, batches(), budget=Budget(1, 3), emit=events.append)
        self.assertEqual(summary['updates'], 1)
        self.assertEqual(summary['skipped_all_ignore'], 1)
        self.assertEqual(summary['stop_reason'], 'max_updates')
        self.assertTrue(optimizer.state)

    def test_all_ignore_batches_do_not_change_model_or_optimizer(self):
        import itertools
        model = CompactUNet()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        batch = torch.zeros((1, 3, 32, 32)), torch.full((1, 32, 32), -100, dtype=torch.int64)
        summary = update_batches(model, optimizer, itertools.repeat(batch),
                                 budget=Budget(1, 2), emit=lambda event: None)
        self.assertEqual(summary['stop_reason'], 'max_batches')
        self.assertEqual(summary['updates'], 0)
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
