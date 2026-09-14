"""Synthetic local-original review, file identity, and recovery checks."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from uavseg.common import AuditError, canonical, sha256
from uavseg.followup import build_page, page_data, validate_records
from uavseg.scenes import pair_id


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = self.root / '.local'
        self.local.mkdir()
        images = []
        for i, key in enumerate(('0001', '0002', '0003')):
            path = self.root / (key + '.png')
            Image.new('RGB', (1024, 1024), (i * 50, 0, 100)).save(path)
            images.append({'id': key, 'image': {'path': 'train/images/' + path.name,
                           'sha256': sha256(path.read_bytes()), 'bytes': path.stat().st_size,
                           'size': [1024, 1024], 'mode': 'RGB'}})
        self.request = {'schema_version': 1, 'kind': 'scene-followup-request',
                        'manifest_sha256': 'a' * 64, 'screen_sha256': 'b' * 64,
                        'source_file_sha256': 'c' * 64, 'required_images': images,
                        'priority_relations': [
                            {'priority': 1, 'original_record': {'left': '0001', 'right': '0002',
                             'pair_id': pair_id('0001', '0002'), 'status': 'confirmed',
                             'reason': 'SECRET_PRIOR_JUDGMENT'}},
                            {'priority': 2, 'left': '0002', 'right': '0003',
                             'pair_id': pair_id('0002', '0003'), 'original_record': None}]}
        self.seal(self.request)
        self.request_path = self.local / 'request.json'
        self.request_path.write_bytes(canonical(self.request))

    @staticmethod
    def seal(value):
        value.pop('request_sha256', None)
        value['request_sha256'] = sha256(canonical(value))

    def record_file(self):
        data = page_data(self.request)
        p = data['pairs'][0]
        return {**data['base'], 'decisions': [{k: p[k] for k in ('left', 'right', 'pair_id')} | {
            'status': 'uncertain', 'reason': '合成图没有可对应的固定地物', 'reviewer': '合成复核人',
            'reviewed_at': '2026-09-14T08:00:00.123Z', 'evidence': 'full_resolution_images',
            'viewed_originals': True,
            'image_sha256': {s: data['images'][p[s]]['sha256'] for s in ('left', 'right')}}]}

    def test_build_without_accessing_images_and_keep_previous_answers_private(self):
        for path in self.root.glob('*.png'):
            path.unlink()  # Only metadata is available on the development terminal.
        output = self.local / 'review.html'
        result = build_page(self.request_path, output)
        page = output.read_text()
        self.assertEqual(result, {'relations': 2, 'images': 3, 'raw_images_read': 0})
        self.assertNotIn('SECRET_PRIOR_JUDGMENT', page)
        self.assertNotIn(str(self.root), page)
        self.assertNotIn('file://', page)
        self.assertIn("connect-src 'none'", page)
        with self.assertRaises(AuditError):
            build_page(self.request_path, output)
        with self.assertRaises(AuditError):
            build_page(self.request_path, self.root / 'public.html')

    def test_request_tamper_and_invalid_relations_fail(self):
        bad = deepcopy(self.request)
        bad['required_images'][0]['image']['sha256'] = 'd' * 64
        with self.assertRaisesRegex(AuditError, '摘要'):
            page_data(bad)
        for mutation in (
            lambda r: r['required_images'][0]['image'].update(path='../0001.png'),
            lambda r: r['priority_relations'].append(r['priority_relations'][0]),
            lambda r: r['priority_relations'][0]['original_record'].update(right='missing'),
        ):
            bad = deepcopy(self.request)
            mutation(bad)
            self.seal(bad)
            with self.assertRaises(AuditError):
                page_data(bad)

    def test_partial_receipt_validation_does_not_merge_or_claim_visual_verification(self):
        result = validate_records(self.request, self.record_file())
        self.assertEqual(result['received'], 1)
        self.assertEqual(result['pending'], 1)
        self.assertFalse(result['decisions_merged'])
        self.assertFalse(result['image_evidence_independently_verified'])

    def test_bad_receipts_are_rejected(self):
        original = self.record_file()
        for key, value in (('reason', ''), ('reviewed_at', '2026-02-30T08:00:00.123Z'),
                           ('reviewed_at', '2026-09-14T08:00:00'), ('viewed_originals', False),
                           ('image_sha256', {'left': 'f' * 64, 'right': 'e' * 64}),
                           ('left', 'unknown'), ('status', 'pending')):
            bad = deepcopy(original)
            bad['decisions'][0][key] = value
            with self.subTest(key=key), self.assertRaises(AuditError):
                validate_records(self.request, bad)
        for key, value in (('kind', 'scene-decisions'), ('request_sha256', 'e' * 64)):
            bad = {**original, key: value}
            with self.assertRaises(AuditError):
                validate_records(self.request, bad)
        bad = deepcopy(original)
        bad['decisions'].append(bad['decisions'][0])
        with self.assertRaises(AuditError):
            validate_records(self.request, bad)

    def test_cli_no_data_config_needed(self):
        output = self.local / 'cli.html'
        result = subprocess.run([sys.executable, '-m', 'uavseg.followup', '--request',
                                 str(self.request_path), '--output', str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['raw_images_read'], 0)

    @unittest.skipUnless(shutil.which('node'), 'Node unavailable: page-script test not run')
    def test_actual_page_file_selection_and_recovery(self):
        output = self.local / 'review.html'
        build_page(self.request_path, output)
        result = subprocess.run(['node', str(Path(__file__).with_name('followup_dom.cjs')),
                                 str(output), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(validate_records(self.request, json.loads(result.stdout))['received'], 1)
