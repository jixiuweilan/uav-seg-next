"""Internal metric edge rules with hand-computed, synthetic expectations."""

import unittest

import numpy as np

from uavseg.common import AuditError
from uavseg.metrics import Confusion, INTERNAL_POLICY


class MetricTests(unittest.TestCase):
    def test_hand_computed_ignore_and_prediction_zero(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        metric.update([[0, 1, 1, 2, 2]], [[8, 1, 0, 1, 2]])
        result = metric.report()
        self.assertAlmostEqual(result['miou'], 5 / 12)
        self.assertEqual(result['valid_pixels'], 4)
        self.assertEqual(result['ignored_pixels'], 1)
        self.assertEqual(result['confusion'][1][0], 1)
        self.assertEqual(sum(result['confusion'][0]), 0)
        self.assertEqual(result['classes'][7]['predicted_pixels'], 0)
        self.assertIsNone(result['classes'][7]['iou'])
        self.assertFalse(result['official_equivalence_confirmed'])

    def test_absent_gt_false_positive_is_zero_not_omitted(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        metric.update([[1, 1]], [[1, 8]])
        self.assertEqual(metric.report()['miou'], .25)
        self.assertEqual(metric.report()['averaged_classes'], 2)
        self.assertEqual(metric.report()['classes'][7]['iou'], 0)

    def test_empty_evaluation_and_all_ignore_are_invalid(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        self.assertIsNone(metric.report()['miou'])
        metric.update([[0, 0]], [[8, 0]])
        result = metric.report()
        self.assertEqual(result['status'], 'invalid_no_valid_pixels')
        self.assertEqual(result['averaged_classes'], 0)
        self.assertEqual(result['images'], 1)
        self.assertIsNone(result['miou'])

    def test_global_counts_and_report_is_an_independent_snapshot(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        metric.update([[1]], [[1]])
        metric.update([[1, 1, 1]], [[0, 0, 0]])
        self.assertEqual(metric.report()['miou'], .25)  # Not image-average .5.
        snapshot = metric.report()
        snapshot['confusion'][1][1] = 100
        self.assertEqual(metric.report()['confusion'][1][1], 1)

    def test_invalid_update_does_not_mutate(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        metric.update([[1]], [[1]])
        before = metric.report()
        for target, pred in (([[1]], [[9]]), ([[-100]], [[1]]), ([[1.]], [[1]]),
                             ([[1]], [[1, 2]]), ([], []), ([[True]], [[1]])):
            with self.assertRaises(AuditError):
                metric.update(target, pred)
            self.assertEqual(metric.report(), before)
        with self.assertRaises(AuditError):
            Confusion(policy='official')

    def test_count_overflow_fails_before_mutation(self):
        metric = Confusion(policy=INTERNAL_POLICY)
        metric.valid_pixels = int(np.iinfo(np.int64).max)
        with self.assertRaises(AuditError):
            metric.update([[1]], [[1]])
        self.assertEqual(metric.images, 0)
