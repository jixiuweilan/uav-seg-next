"""Group constraints and draft-split provenance on the existing synthetic fixture."""

from copy import deepcopy
import json
import shutil
import subprocess
import sys
import unittest

import test_scenes
from uavseg.common import AuditError, canonical
from uavseg.splits import propose_split


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_scenes.SceneTests('test_confirmed_cross_split_group_is_detected')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_complete_disjoint_and_no_confirmed_or_exact_component_split(self):
        f = self.fixture
        decisions = f.decisions(f.decision('a', 'b'), f.decision('b', 'c'))
        proposal = propose_split(f.document, f.screen, decisions)
        split = proposal['split']
        train, val = set(split['train_ids']), set(split['val_ids'])
        self.assertFalse(train & val)
        self.assertEqual(train | val, set('abcdefgh'))
        self.assertEqual(len(val), 1)
        # h is an exact decoded duplicate of a, with no manual a/h edge.
        self.assertTrue(set('abch') <= train or set('abch') <= val)
        self.assertFalse(split['frozen'] or split['scene_independence_certified'] or split['training_authorized'])
        self.assertGreater(split['unreviewed_relations'], 0)
        self.assertEqual(split['class_coverage']['val']['pixels_by_class'][1], 1024 * 1024)
        self.assertEqual(split['class_coverage']['train']['missing_evaluated_classes'], list(range(2, 9)))
        self.assertEqual(proposal, propose_split(f.document, f.screen, decisions))
        reversed_decisions = f.decisions(*reversed(decisions['decisions']))
        again = propose_split(f.document, f.screen, reversed_decisions)['split']
        self.assertEqual(again['train_ids'], split['train_ids'])
        self.assertEqual(again['val_ids'], split['val_ids'])

    def test_rejected_confirmed_chain_and_exact_duplicate_contradictions_fail(self):
        f = self.fixture
        for rows in ((f.decision('a', 'b'), f.decision('b', 'c'), f.decision('a', 'c', 'rejected')),
                     (f.decision('a', 'h', 'rejected'),)):
            with self.assertRaisesRegex(AuditError, '矛盾'):
                propose_split(f.document, f.screen, f.decisions(*rows))

    def test_invalid_target_and_tampered_manifest_fail(self):
        f = self.fixture
        for target in (0, 8, 1.5, True):
            with self.assertRaises(AuditError):
                propose_split(f.document, f.screen, f.decisions(), validation_count=target)
        bad = deepcopy(f.document)
        bad['manifest']['train'][0]['mask']['class_counts'][1] += 1
        with self.assertRaisesRegex(AuditError, '摘要'):
            propose_split(bad, f.screen, f.decisions())

    def test_whole_dataset_component_cannot_be_forced_into_two_splits(self):
        f = self.fixture
        decisions = f.decisions(*(f.decision(a, b) for a, b in zip('abcdefg', 'bcdefgh')))
        with self.assertRaisesRegex(AuditError, '非空'):
            propose_split(f.document, f.screen, decisions)

    def test_cli_uses_only_json_and_does_not_overwrite(self):
        f = self.fixture
        values = {'manifest': f.document, 'screen': f.screen, 'decisions': f.decisions(f.decision('a', 'b'))}
        args = [sys.executable, '-m', 'uavseg.splits']
        for name, value in values.items():
            path = f.local / (name + '.json')
            path.write_bytes(canonical(value))
            args += ['--' + name, str(path)]
        args += ['--output', str(f.local / 'candidate.json')]
        shutil.rmtree(f.raw)  # Only audited JSON is available on the code terminal.
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'draft')
        again = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(again.returncode, 2)
