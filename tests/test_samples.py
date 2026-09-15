"""All file reads in these tests use newly generated synthetic fixtures."""

from copy import deepcopy
import shutil
import unittest

import numpy as np

import test_scenes
from uavseg.common import AuditError, canonical, sha256
from uavseg.data import Geometry
from uavseg.samples import AuditedSamples, candidate_ids
from uavseg.splits import propose_split


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.f = test_scenes.SceneTests('test_confirmed_cross_split_group_is_detected')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.proposal = propose_split(self.f.document, self.f.screen, self.f.decisions())

    def test_train_crop_and_validation_full_image(self):
        train = AuditedSamples(self.f.raw, self.f.document, partition='train', proposal=self.proposal)
        sample = train.read(train.ids[0], geometry=Geometry(1, 2, 512, 512, quarter_turns=1))
        self.assertEqual(sample['image'].shape, (3, 512, 512))
        self.assertTrue(np.all(sample['target'] == 0))
        self.assertEqual(sample['valid_pixels'], 512 * 512)
        val = AuditedSamples(self.f.raw, self.f.document, partition='val', proposal=self.proposal)
        self.assertEqual(val.read(val.ids[0])['image'].shape, (3, 1024, 1024))
        with self.assertRaises(AuditError):
            val.read(val.ids[0], geometry=Geometry(0, 0, 512, 512))
        with self.assertRaises(AuditError):
            train.read(val.ids[0])

    def test_prediction_needs_no_training_files_or_masks(self):
        shutil.rmtree(self.f.raw / 'train')
        samples = AuditedSamples(self.f.raw, self.f.document, partition='test')
        result = samples.read('test')
        self.assertEqual(set(result), {'id', 'image'})
        self.assertEqual(result['image'].dtype, np.float32)
        np.testing.assert_allclose(result['image'][:, 0, 0], np.array([1, 2, 3]) / 255)

    def test_changed_bytes_and_symlinks_fail(self):
        samples = AuditedSamples(self.f.raw, self.f.document, partition='test')
        path = self.f.raw / 'test/images/test.png'
        before = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(AuditError, '身份'):
            samples.read('test')
        path.unlink()
        outside = self.f.local / 'external.png'
        outside.write_bytes(before)
        path.symlink_to(outside)
        with self.assertRaisesRegex(AuditError, '符号链接'):
            samples.read('test')

    def test_mutations_after_construction_do_not_change_reads(self):
        document = deepcopy(self.f.document)
        samples = AuditedSamples(self.f.raw, document, partition='test')
        document['manifest']['test'][0]['image']['sha256'] = '0' * 64
        self.assertEqual(samples.read('test')['id'], 'test')

    def test_rehashed_bad_paths_and_decode_statistics_fail(self):
        for key, value in (('path', '../test.png'), ('path', '/test.png'),
                           ('path', 'test/images/other.png'), ('pixel_sha256', '0' * 64)):
            document = deepcopy(self.f.document)
            document['manifest']['test'][0]['image'][key] = value
            document['manifest_sha256'] = sha256(canonical(document['manifest']))
            with self.assertRaises(AuditError):
                AuditedSamples(self.f.raw, document, partition='test').read('test')

    def test_split_tampering_and_rehashed_overlap_fail(self):
        bad = deepcopy(self.proposal)
        bad['split']['train_ids'].append(bad['split']['val_ids'][0])
        with self.assertRaisesRegex(AuditError, '摘要'):
            candidate_ids(self.f.document, bad, 'train')
        bad['split']['train_ids'].sort()
        bad['split_sha256'] = sha256(canonical(bad['split']))
        with self.assertRaisesRegex(AuditError, '互斥'):
            candidate_ids(self.f.document, bad, 'train')
