"""The generated plan binds inputs while refusing to authorize training."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import test_scenes
from uavseg.common import AuditError, canonical, sha256
from uavseg.runplan import draft_plan
from uavseg.splits import propose_split
from uavseg.training import Budget


class RunPlanTests(unittest.TestCase):
    def setUp(self):
        self.f = test_scenes.SceneTests('test_confirmed_cross_split_group_is_detected')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.proposal = propose_split(self.f.document, self.f.screen, self.f.decisions())

    def test_identity_bound_draft_has_all_hard_gates(self):
        result = draft_plan(self.f.document, self.proposal, source_sha256=sha256(b'code'),
                            batch_size=2, budget=Budget(3, 5), validation_cadence=1)
        plan = result['plan']
        self.assertEqual(result['plan_sha256'], sha256(canonical(plan)))
        self.assertFalse(plan['training_authorized'])
        self.assertFalse(plan['formal_training_command_available'])
        self.assertFalse(plan['data']['split_frozen'])
        self.assertFalse(plan['metric']['experimental_selection_allowed'])
        self.assertEqual(plan['data']['train_samples'] + plan['data']['validation_samples'], 8)
        self.assertEqual(set(plan['unresolved_gates']), {'frozen_split', 'official_metric_edges',
                         'execution_resource_budget', 'owner_training_authorization'})

    def test_bad_runtime_choices_and_tampered_split_fail(self):
        base = {'source_sha256': sha256(b'code'), 'batch_size': 1,
                'budget': Budget(2, 3), 'validation_cadence': 1}
        for key, value in (('source_sha256', 'bad'), ('batch_size', 0),
                           ('budget', object()), ('validation_cadence', 0),
                           ('validation_cadence', 3)):
            with self.assertRaises(AuditError):
                draft_plan(self.f.document, self.proposal, **{**base, key: value})
        bad = deepcopy(self.proposal)
        bad['split']['frozen'] = True
        with self.assertRaisesRegex(AuditError, '摘要'):
            draft_plan(self.f.document, bad, **base)
        bad['split_sha256'] = sha256(canonical(bad['split']))
        with self.assertRaisesRegex(AuditError, '未冻结'):
            draft_plan(self.f.document, bad, **base)

    def test_cli_reads_json_only_and_never_overwrites(self):
        manifest = self.f.local / 'manifest.json'
        split = self.f.local / 'split.json'
        project_local = Path(__file__).resolve().parent.parent / '.local'
        project_local.mkdir(exist_ok=True)
        manifest.write_bytes(canonical(self.f.document))
        split.write_bytes(canonical(self.proposal))
        shutil.rmtree(self.f.raw)
        with tempfile.TemporaryDirectory(dir=project_local) as temporary:
            output = Path(temporary) / 'run-plan.json'
            args = [sys.executable, '-m', 'uavseg.runplan', '--manifest', str(manifest),
                    '--split', str(split), '--source-sha256', sha256(b'code'), '--batch-size', '1',
                    '--max-updates', '2', '--max-batches', '3', '--validation-cadence', '1',
                    '--output', str(output)]
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(json.loads(run.stdout)['training_authorized'])
            self.assertEqual(json.loads(output.read_bytes())['plan']['data']['train_samples'], 7)
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)
