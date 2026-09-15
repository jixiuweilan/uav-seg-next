"""Validation traversal and earliest-best rules use hand-sized arrays."""

import unittest

import numpy as np

from uavseg.common import AuditError, sha256
from uavseg.metrics import INTERNAL_POLICY
from uavseg.validation import EarliestBest, SelectionContext, evaluate_samples


class ValidationTests(unittest.TestCase):
    def test_exact_order_global_metric_and_events(self):
        samples = iter(({'id': 'a', 'target': np.array([[1, 1]], np.uint8)},
                        {'id': 'b', 'target': np.array([[1, 1, 1]], np.uint8)}))
        predictions = {'a': np.array([[1, 1]], np.uint8),
                       'b': np.array([[0, 0, 0]], np.uint8)}
        events = []
        report = evaluate_samples(samples, lambda row: predictions[row['id']],
                                  expected_ids=('a', 'b'), emit=events.append)
        self.assertEqual(report['miou'], .4)
        self.assertEqual(report['images'], 2)
        self.assertEqual([event['event'] for event in events],
                         ['validation_started', 'validation_sample_finished',
                          'validation_sample_finished', 'validation_finished'])

    def test_missing_extra_reordered_and_unlabeled_samples_fail(self):
        cases = ([{'id': 'a', 'target': [[1]]}],
                 [{'id': 'b', 'target': [[1]]}, {'id': 'a', 'target': [[1]]}],
                 [{'id': 'a', 'target': [[1]]}, {'id': 'b', 'target': [[1]]},
                  {'id': 'c', 'target': [[1]]}],
                 [{'id': 'a', 'target': [[1]]}, {'id': 'b', 'target': [[1]]}, None],
                 [{'id': 'a'}, {'id': 'b', 'target': [[1]]}])
        for samples in cases:
            events = []
            with self.assertRaises(AuditError):
                evaluate_samples(samples, lambda row: row['target'],
                                 expected_ids=('a', 'b'), emit=events.append)
            self.assertIn(events[-1]['event'], ('validation_failed', 'validation_interrupted'))

    def test_no_valid_pixels_and_bad_expected_contract_fail(self):
        with self.assertRaisesRegex(AuditError, '没有有效'):
            evaluate_samples([{'id': 'a', 'target': [[0]]}], lambda row: [[0]],
                             expected_ids=('a',), emit=lambda event: None)
        for ids in ([], ['a'], ('b', 'a'), ('a', 'a')):
            with self.assertRaises(AuditError):
                evaluate_samples([], lambda row: row, expected_ids=ids, emit=lambda event: None)

    def test_earliest_tie_wins_and_strict_improvement_replaces(self):
        context = SelectionContext(sha256(b'manifest'), sha256(b'split'), 10)
        selector = EarliestBest(context)
        def metric(score):
            return {'policy': INTERNAL_POLICY, 'official_equivalence_confirmed': False,
                    'status': 'valid', 'miou': score, 'averaged_classes': 8}
        first = selector.consider(updates=10, report=metric(.5), checkpoint_sha256=sha256(b'a'))
        tied = selector.consider(updates=20, report=metric(.5), checkpoint_sha256=sha256(b'b'))
        better = selector.consider(updates=30, report=metric(.6), checkpoint_sha256=sha256(b'c'))
        self.assertTrue(first['selected'] and better['selected'])
        self.assertFalse(tied['selected'])
        self.assertEqual(selector.report()['best']['updates'], 30)
        report = selector.report()
        report['records'][0]['miou'] = 0
        self.assertEqual(selector.report()['records'][0]['miou'], .5)
        self.assertFalse(selector.report()['official_model_selection'])

    def test_selector_rejects_off_cadence_replay_and_bad_metrics(self):
        selector = EarliestBest(SelectionContext(sha256(b'm'), sha256(b's'), 10))
        valid = {'policy': INTERNAL_POLICY, 'official_equivalence_confirmed': False,
                 'status': 'valid', 'miou': .5, 'averaged_classes': 8}
        selector.consider(updates=10, report=valid, checkpoint_sha256=sha256(b'a'))
        for updates, report, digest in ((15, valid, sha256(b'b')), (10, valid, sha256(b'b')),
                                        (20, {**valid, 'miou': float('nan')}, sha256(b'b')),
                                        (20, {**valid, 'miou': True}, sha256(b'b')),
                                        (20, {**valid, 'official_equivalence_confirmed': True}, sha256(b'b')),
                                        (20, valid, 'bad')):
            with self.assertRaises(AuditError):
                selector.consider(updates=updates, report=report, checkpoint_sha256=digest)
        for args in ((sha256(b'm'), sha256(b's'), 0), ('bad', sha256(b's'), 1)):
            with self.assertRaises(AuditError):
                SelectionContext(*args)
