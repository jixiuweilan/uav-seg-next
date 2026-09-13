"""Synthetic scene candidates and human-review contracts; no real decisions."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from uavseg.audit import audit_data
from uavseg.common import AuditError, DataPaths, canonical, sha256
from uavseg.review import empty_decisions, group_report, review_page
from uavseg.scenes import extract_features, pair_id, screen_features, validate_screen


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw = self.base / 'raw'
        self.local = self.base / '.local'
        self.local.mkdir()
        for folder in ('train/images', 'train/masks', 'test/images'):
            (self.raw / folder).mkdir(parents=True)
        rng = np.random.default_rng(71)
        a = rng.integers(20, 200, (32, 32, 3), dtype=np.uint8).repeat(32, 0).repeat(32, 1)
        other = rng.integers(20, 200, (32, 32, 3), dtype=np.uint8).repeat(32, 0).repeat(32, 1)
        arrays = {'a': a, 'b': np.rot90(a), 'c': a + 5,
                  'd': np.rot90(np.fliplr(a)), 'e': other,
                  'f': np.full_like(a, 60), 'g': np.full_like(a, 90), 'h': a}
        for name, array in arrays.items():
            Image.fromarray(array).save(self.raw / f'train/images/{name}.png', compress_level=0 if name == 'h' else 6)
            Image.new('L', (1024, 1024), 1).save(self.raw / f'train/masks/{name}.png')
        Image.new('RGB', (1024, 1024), (1, 2, 3)).save(self.raw / 'test/images/test.png')
        self.config = self.local / 'paths.json'
        self.config.write_bytes(canonical({'dataset_root': '../raw', 'access': 'read-only',
                                          'train_images': 'train/images', 'train_masks': 'train/masks',
                                          'test_images': 'test/images'}))
        self.paths = DataPaths(self.config)
        split_paths = {}
        for key, values in (('train', 'a\nc\nd\ne\nf\ng\nh\n'), ('val', 'b\n'),
                            ('train_all', '\n'.join(arrays) + '\n')):
            path = self.local / (key + '.txt')
            path.write_text(values)
            split_paths[key] = path
        self.document = audit_data(self.paths, split_paths)
        self.features = extract_features(self.paths, self.document)
        self.screen = screen_features(self.features, self.document, {'controls': 3})

    def decision(self, left, right, status='confirmed'):
        key = pair_id(left, right)
        return {'pair_id': key, 'left': left, 'right': right, 'status': status,
                'reviewer': 'Synthetic Reviewer', 'reviewed_at': '2026-09-09T01:00:00+00:00',
                'reason': 'Synthetic fixture reviewed at original resolution.',
                'evidence': 'full_resolution_images',
                'origin': 'screen' if any(p['pair_id'] == key for p in self.screen['screen']['pairs']) else 'manual'}

    def decisions(self, *records):
        return {**empty_decisions(self.document, self.screen), 'decisions': list(records)}

    def test_rotations_mirrors_brightness_and_exact_pixels_are_candidates(self):
        pairs = {(r['left'], r['right']): r for r in self.screen['screen']['pairs'] if r['queue'] == 'candidate'}
        for key in (('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'h')):
            self.assertIn(key, pairs)
        self.assertEqual(pairs['a', 'b']['pose'], 'r270')
        self.assertTrue(pairs['a', 'h']['exact_pixels'])
        self.assertFalse(pairs['a', 'b']['exact_pixels'])
        self.assertNotIn(('a', 'e'), pairs)

    def test_low_texture_is_flagged_not_deleted_or_grouped(self):
        self.assertEqual(self.screen['screen']['low_texture_ids'], ['f', 'g'])
        self.assertEqual(self.screen['screen']['train_images'], 8)
        self.assertFalse(any(p['left'] == 'f' and p['right'] == 'g' and p['queue'] == 'candidate'
                             for p in self.screen['screen']['pairs']))
        result = group_report(self.document, self.screen)['report']
        self.assertEqual(result['groups'], [])
        self.assertEqual(len(result['ungrouped_ids']), 8)
        self.assertFalse(result['scene_independence_certified'])

    def test_deterministic_bounded_queues_and_controls(self):
        first = screen_features(self.features, self.document, {'max_per_partition': 1, 'controls': 3})
        second = screen_features(self.features, self.document, {'max_per_partition': 1, 'controls': 3})
        self.assertEqual(canonical(first), canonical(second))
        payload = first['screen']
        self.assertEqual(payload['total_pairs'], 28)
        self.assertEqual(payload['retained_by_partition']['cross_split'], 1)
        self.assertEqual(payload['retained_by_partition']['within_split'], 1)
        self.assertGreater(sum(payload['omitted_by_partition'].values()), 0)
        self.assertEqual(payload['controls_retained'], 3)
        all_hits = {p['pair_id'] for p in self.screen['screen']['pairs'] if p['queue'] == 'candidate'}
        controls = {p['pair_id'] for p in payload['pairs'] if p['queue'] == 'control'}
        self.assertFalse(all_hits & controls)

    def test_bad_parameters_and_changed_data_fail(self):
        for config in ({'hash_max': 65}, {'rms_max': float('nan')}, {'max_per_partition': 0}):
            with self.subTest(config=config), self.assertRaises(AuditError):
                screen_features(self.features, self.document, config)
        path = self.raw / 'train/images/a.png'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(AuditError, 'a: image changed'):
            extract_features(self.paths, self.document)

    def test_stale_or_malformed_candidate_document_fails(self):
        wrong = deepcopy(self.screen)
        wrong['screen']['manifest_sha256'] = 'wrong'
        wrong['screen_sha256'] = sha256(canonical(wrong['screen']))
        with self.assertRaisesRegex(AuditError, 'different manifest'):
            validate_screen(wrong, self.document)
        wrong = deepcopy(self.screen)
        wrong['screen']['pairs'][0]['rms'] = '<script>'
        wrong['screen_sha256'] = sha256(canonical(wrong['screen']))
        with self.assertRaisesRegex(AuditError, 'invalid candidate score'):
            validate_screen(wrong, self.document)

    def test_confirmed_cross_split_group_is_detected(self):
        result = group_report(self.document, self.screen, self.decisions(self.decision('a', 'b')))['report']
        self.assertEqual(result['status'], 'known_split_conflict')
        self.assertEqual(result['groups'][0]['members'], ['a', 'b'])
        self.assertEqual(len(result['cross_split_group_ids']), 1)
        self.assertFalse(result['scene_independence_certified'])

    def test_rejected_and_uncertain_decisions_do_not_create_edges(self):
        for status in ('rejected', 'uncertain'):
            result = group_report(self.document, self.screen, self.decisions(self.decision('a', 'b', status)))['report']
            self.assertEqual(result['groups'], [])
            self.assertEqual(result['cross_split_group_ids'], [])
            self.assertFalse(result['reviewer_identity_authenticated'])

    def test_transitive_groups_and_contradictory_rejections(self):
        decisions = self.decisions(self.decision('a', 'c'), self.decision('c', 'e'),
                                   self.decision('a', 'e', 'rejected'))
        result = group_report(self.document, self.screen, decisions)['report']
        self.assertEqual(result['groups'][0]['members'], ['a', 'c', 'e'])
        self.assertEqual(result['status'], 'inconsistent_review')
        self.assertEqual(result['contradictory_rejections'], [pair_id('a', 'e')])

    def test_decision_identity_and_required_provenance(self):
        for field, value in (('reviewer', ''), ('reason', ''), ('reviewed_at', '2026-09-09'),
                             ('evidence', 'thumbnail_only'), ('status', 'approve'), ('origin', 'manual')):
            decision = self.decision('a', 'b')
            decision[field] = value
            with self.subTest(field=field), self.assertRaises(AuditError):
                group_report(self.document, self.screen, self.decisions(decision))
        decision = self.decision('a', 'b')
        with self.assertRaisesRegex(AuditError, 'duplicate'):
            group_report(self.document, self.screen, self.decisions(decision, decision))
        stale = self.decisions(decision)
        stale['screen_sha256'] = 'wrong'
        with self.assertRaisesRegex(AuditError, 'identity mismatch'):
            group_report(self.document, self.screen, stale)

    def test_local_review_page_and_actual_js_export_import(self):
        page = self.local / 'review.html'
        review_page(self.paths, self.document, self.screen, page, [self.raw, self.config])
        content = page.read_text()
        self.assertIn('data:image/jpeg;base64,', content)
        self.assertIn('file://', content)
        self.assertNotIn('src="https://', content)
        self.assertNotIn('fetch(', content)
        self.assertIn('Record a relation absent from the queue', content)
        if not shutil.which('node'):
            self.skipTest('Node unavailable; browser-script harness not run')
        result = subprocess.run(['node', 'tests/review_dom.cjs', str(page)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        decisions = json.loads(result.stdout)
        self.assertEqual(len(decisions['decisions']), 2)
        self.assertEqual(decisions['decisions'][0]['reviewer'], 'Synthetic Reviewer')
        self.assertEqual(sum(row['origin'] == 'manual' for row in decisions['decisions']), 1)
        report = group_report(self.document, self.screen, decisions)['report']
        self.assertEqual(len(report['groups']), 1)
        self.assertEqual(report['review_counts']['manual']['rejected'], 1)

    def test_review_output_protection_and_atomic_failure(self):
        with self.assertRaisesRegex(AuditError, 'under .local'):
            review_page(self.paths, self.document, self.screen, self.base / 'review.html', [self.raw])
        output = self.local / 'bad.html'
        (self.raw / 'train/images/a.png').write_bytes(b'changed')
        with self.assertRaisesRegex(AuditError, 'changed since audit'):
            review_page(self.paths, self.document, self.screen, output, [self.raw])
        self.assertFalse(output.exists())
        self.assertEqual(list(self.local.glob('.uavseg-*')), [])

    def test_cli_screen_review_and_pending_or_conflicting_group_report(self):
        manifest = self.local / 'audit.json'
        manifest.write_bytes(canonical(self.document))
        candidates = self.local / 'candidates.json'
        def run(command, output, *args):
            return subprocess.run([sys.executable, '-m', 'uavseg', command,
                                   '--config', str(self.config), '--manifest', str(manifest),
                                   '--output', str(output), *map(str, args)], capture_output=True, text=True)
        result = run('screen-scenes', candidates, '--controls', '3')
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run('review-scenes', self.local / 'cli-review.html', '--screen', candidates)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run('check-groups', self.local / 'pending.json', '--screen', candidates)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'review_pending')
        decisions = self.local / 'decisions.json'
        self.assertEqual(json.loads(candidates.read_text())['screen_sha256'], self.screen['screen_sha256'])
        decisions.write_bytes(canonical(self.decisions(self.decision('a', 'b'))))
        result = run('check-groups', self.local / 'conflict.json', '--screen', candidates, '--decisions', decisions)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'known_split_conflict')
        self.assertTrue((self.local / 'conflict.json').is_file())


if __name__ == '__main__':
    unittest.main()
