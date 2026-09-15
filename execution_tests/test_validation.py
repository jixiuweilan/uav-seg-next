"""Execution-import adapter checks without official data or model computation."""

import unittest
from unittest.mock import patch

import numpy as np

from uavseg.common import AuditError
from uavseg.runtime import validate_model


class ValidationAdapterTests(unittest.TestCase):
    def test_model_targets_convert_to_official_ids_without_mutation(self):
        target = np.array([[-100, 0, 7]], dtype=np.int64)
        sample = {'id': 'a', 'image': np.zeros((3, 2, 3), np.float32), 'target': target}
        with patch('uavseg.runtime.predict_image', return_value=np.array([[8, 1, 8]], np.uint8)) as predict:
            report = validate_model(object(), [sample], expected_ids=('a',), emit=lambda event: None)
        np.testing.assert_array_equal(target, [[-100, 0, 7]])
        self.assertEqual(report['valid_pixels'], 2)
        self.assertEqual(report['ignored_pixels'], 1)
        self.assertEqual(report['miou'], 1)
        self.assertEqual(predict.call_args.kwargs['device'], 'cpu')

    def test_invalid_model_target_fails_before_prediction(self):
        sample = {'id': 'a', 'image': np.zeros((3, 2, 2), np.float32),
                  'target': np.zeros((2, 2), np.float32)}
        with patch('uavseg.runtime.predict_image') as predict:
            with self.assertRaises(AuditError):
                validate_model(object(), [sample], expected_ids=('a',), emit=lambda event: None)
        predict.assert_not_called()
