"""Numerical tests for the execution machine only; no dataset or CUDA access."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from uavseg.check_baseline import source_identity
from uavseg.common import AuditError, canonical, sha256
from uavseg.data import prepare_pair
from uavseg.images import inspect_file
from uavseg.metrics import Confusion, INTERNAL_POLICY
from uavseg.model import CompactUNet
from uavseg.runtime import load_checkpoint, predict_image, save_checkpoint, update_batch
from uavseg.samples import AuditedSamples
from uavseg.submission import pack_submission, validate_zip


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CompactUNet()

    def test_odd_shapes_and_seed_replay(self):
        torch.manual_seed(0)
        other = CompactUNet()
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]))
        self.model.eval()
        with torch.inference_mode():
            for shape in ((16, 16), (33, 35), (512, 512)):
                output = self.model(torch.zeros((1, 3, *shape)))
                self.assertEqual(output.shape, (1, 8, *shape))
                self.assertTrue(torch.isfinite(output).all().item())
        with self.assertRaises(AuditError):
            self.model(torch.zeros((1, 3, 15, 32)))

    def test_real_update_and_all_ignore_leaves_optimizer_unchanged(self):
        labels = np.ones((32, 32), dtype=np.uint8)
        labels[0] = 0
        labels[1] = 8
        sample = prepare_pair(np.full((32, 32, 3), 127, np.uint8), labels)
        images = torch.from_numpy(sample['image']).unsqueeze(0)
        targets = torch.from_numpy(sample['target']).unsqueeze(0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.0001)
        before = self.model.head.weight.detach().clone()
        result = update_batch(self.model, optimizer, images, targets)
        self.assertTrue(result['updated'])
        self.assertEqual(result['valid_pixels'], 31 * 32)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertFalse(torch.equal(before, self.model.head.weight))
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        state = deepcopy(optimizer.state_dict())
        result = update_batch(self.model, optimizer, images, torch.full_like(targets, -100))
        self.assertFalse(result['updated'])
        self.assertIsNone(result['loss'])
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(weights[key], value))
        after = optimizer.state_dict()
        self.assertEqual(state['param_groups'], after['param_groups'])
        for index, values in state['state'].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, after['state'][index][key]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_invalid_inputs_and_nonfinite_gradient_do_not_step(self):
        images = torch.zeros((1, 3, 32, 32))
        targets = torch.zeros((1, 32, 32), dtype=torch.int64)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        for image, target in ((images + float('nan'), targets), (images + 2, targets),
                              (images, targets + 8), (images, targets.float())):
            with self.assertRaises(AuditError):
                update_batch(self.model, optimizer, image, target)
        hook = self.model.head.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        try:
            with self.assertRaisesRegex(AuditError, '梯度'):
                update_batch(self.model, optimizer, images, targets)
        finally:
            hook.remove()
        self.assertFalse(optimizer.state)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(weights[key], value))

    def test_checkpoint_roundtrip_and_identity_rejections(self):
        provenance = {'manifest_sha256': sha256(b'synthetic-manifest'),
                      'split_sha256': sha256(b'synthetic-split'), 'source_sha256': source_identity(),
                      'metric_policy': INTERNAL_POLICY, 'seed': 0, 'purpose': 'synthetic-runtime-check'}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'weights.pt'
            receipt = save_checkpoint(self.model, path, provenance=provenance, protected=[])
            rng_state = torch.get_rng_state().clone()
            restored = load_checkpoint(path, expected_sha256=receipt['checkpoint_sha256'], provenance=provenance)
            self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
            self.model.eval()
            with torch.inference_mode():
                image = torch.zeros((1, 3, 33, 35))
                self.assertTrue(torch.equal(self.model(image), restored(image)))
            for expected_hash, source in (('0' * 64, provenance),
                                          (receipt['checkpoint_sha256'], {**provenance, 'seed': 1})):
                with self.assertRaises(AuditError):
                    load_checkpoint(path, expected_sha256=expected_hash, provenance=source)
            with self.assertRaises(AuditError):
                save_checkpoint(self.model, path, provenance=provenance, protected=[])
            with self.assertRaises(AuditError):
                save_checkpoint(self.model, Path(temporary) / 'formal.pt',
                                provenance={**provenance, 'purpose': 'formal-training'}, protected=[])
            with self.assertRaises(AuditError):
                save_checkpoint(self.model, Path(temporary) / 'protected.pt',
                                provenance=provenance, protected=[Path(temporary)])

    def test_full_image_to_png_zip_without_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            raw.mkdir()
            source = raw / 'example.png'
            Image.new('RGB', (1024, 1024), (32, 64, 128)).save(source)
            metadata = {**inspect_file(source), 'path': source.name}
            manifest = {'schema_version': 1, 'kind': 'official-data-audit', 'train': [],
                        'test': [{'id': 'example', 'image': metadata}]}
            document = {'manifest': manifest, 'manifest_sha256': sha256(canonical(manifest))}
            sample = AuditedSamples(raw, document, partition='test').read('example')
            self.model.train()
            prediction = predict_image(self.model, sample['image'])
            self.assertTrue(self.model.training)
            self.assertEqual(prediction.dtype, np.uint8)
            self.assertGreaterEqual(int(prediction.min()), 1)
            self.assertLessEqual(int(prediction.max()), 8)
            metric = Confusion(policy=INTERNAL_POLICY)
            metric.update(prediction, prediction)
            self.assertEqual(metric.report()['miou'], 1)
            output = root / 'predictions'
            output.mkdir()
            Image.fromarray(prediction).save(output / 'example.png')
            archive = root / 'submission.zip'
            report = pack_submission(output, archive, document, protected=[raw])
            self.assertTrue(report['valid'])
            self.assertEqual(validate_zip(archive, document)['files'], 1)
            self.assertEqual(inspect_file(source), {k: v for k, v in metadata.items() if k != 'path'})
