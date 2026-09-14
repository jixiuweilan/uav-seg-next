"""Hand-checkable geometry and label mapping, without a training runtime."""

import unittest

import numpy as np

from uavseg.common import AuditError
from uavseg.data import Geometry, decode_predictions, encode_targets, prepare_pair
from uavseg.data import sample_geometry, transform_pair


class DataTests(unittest.TestCase):
    def test_all_label_ids_and_invalid_values(self):
        mask = np.arange(9, dtype=np.uint8)[None, :]
        np.testing.assert_array_equal(encode_targets(mask), [[-100, 0, 1, 2, 3, 4, 5, 6, 7]])
        np.testing.assert_array_equal(decode_predictions(encode_targets(mask)[:, 1:]), mask[:, 1:])
        for value in (np.array([[9]], dtype=np.uint8), mask.astype(float), np.zeros((0, 1), np.uint8)):
            with self.assertRaises(AuditError):
                encode_targets(value)
        for value in (np.array([[-100]]), np.array([[8]]), np.array([[1.0]])):
            with self.assertRaises(AuditError):
                decode_predictions(value)

    def test_odd_geometry_hand_computed_alignment_and_input_preservation(self):
        mask = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 8, 0]], dtype=np.uint8)
        image = np.stack((mask, mask + 10, mask + 20), axis=2)
        before = image.copy(), mask.copy()
        geometry = Geometry(1, 1, 2, 3, horizontal_flip=True, quarter_turns=1)
        pixels, labels = transform_pair(image, mask, geometry)
        expected = np.array([[1, 4], [2, 5], [3, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(labels, expected)
        np.testing.assert_array_equal(pixels[:, :, 0], expected)
        np.testing.assert_array_equal(pixels[:, :, 1], expected + 10)
        self.assertTrue(labels.flags.c_contiguous and pixels.flags.c_contiguous)
        labels[:] = 0
        pixels[:] = 0
        np.testing.assert_array_equal(image, before[0])
        np.testing.assert_array_equal(mask, before[1])

    def test_one_pixel_object_survives_all_flips_and_turns(self):
        mask = np.zeros((5, 7), np.uint8)
        mask[1, 4] = 8
        image = np.repeat(mask[:, :, None], 3, axis=2)
        for h in (False, True):
            for v in (False, True):
                for k in range(4):
                    pixels, labels = transform_pair(image, mask, Geometry(0, 0, 5, 7, h, v, k))
                    self.assertEqual(np.count_nonzero(labels == 8), 1)
                    np.testing.assert_array_equal(pixels[:, :, 0], labels)
                    self.assertEqual(set(np.unique(labels)), {0, 8})

    def test_seeded_draw_replay_and_crop_boundaries(self):
        one, two = np.random.default_rng(0), np.random.default_rng(0)
        first = [sample_geometry((9, 11), (5, 7), one) for _ in range(30)]
        second = [sample_geometry((9, 11), (5, 7), two) for _ in range(30)]
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= g.top <= 4 and 0 <= g.left <= 4 for g in first))
        full = sample_geometry((9, 11), (9, 11), one)
        self.assertEqual((full.top, full.left), (0, 0))
        for crop in ((10, 7), (0, 7), (5.0, 7)):
            with self.assertRaises(AuditError):
                sample_geometry((9, 11), crop, one)

    def test_full_validation_normalization_and_all_ignore_retained(self):
        mask = np.zeros((3, 5), np.uint8)
        image = np.full((3, 5, 3), 255, np.uint8)
        result = prepare_pair(image, mask)
        self.assertEqual(result['image'].shape, (3, 3, 5))
        self.assertEqual(result['image'].dtype, np.float32)
        self.assertTrue(np.all(result['image'] == 1))
        self.assertTrue(np.all(result['target'] == -100))
        self.assertEqual(result['valid_pixels'], 0)
        self.assertEqual(result['geometry']['quarter_turns'], 0)
        self.assertTrue(np.all(image == 255))

    def test_bad_geometry_and_mismatched_pair_fail(self):
        image, mask = np.zeros((3, 5, 3), np.uint8), np.zeros((3, 5), np.uint8)
        for g in (Geometry(-1, 0, 2, 2), Geometry(0, 4, 2, 2), Geometry(0, 0, 3, 5, quarter_turns=4),
                  Geometry(0, 0, 3, 5, horizontal_flip=1)):
            with self.assertRaises(AuditError):
                transform_pair(image, mask, g)
        with self.assertRaises(AuditError):
            prepare_pair(image[:, :-1], mask)
