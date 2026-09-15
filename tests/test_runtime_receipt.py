"""An attested-looking summary must not replace report and source checks."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uavseg.common import AuditError, canonical, sha256
from uavseg.runtime_receipt import inspect_receipt, main, revision_identity


class RuntimeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.report = {'schema_version': 1, 'kind': 'synthetic-baseline-runtime-check',
                       'source_sha256': 'a' * 64, 'device': 'cpu', 'seed': 0,
                       'official_data_read': False, 'formal_training_started': False,
                       'suite': 'all', 'tests_run': 5, 'status': 'passed',
                       'python_version': '3.11.15', 'torch_version': '2.8.0+cu128',
                       'numpy_version': '2.0.2', 'pillow_version': '11.3.0',
                       'failures': [], 'errors': [], 'skipped': []}

    def test_full_report_retains_unverified_execution_boundary(self):
        result = inspect_receipt(self.report, expected_source='a' * 64)
        self.assertTrue(result['full_suite_reported_passed'])
        self.assertFalse(result['execution_independently_observed'])
        self.assertFalse(result['reporter_identity_authenticated'])
        legacy = deepcopy(self.report)
        del legacy['suite']
        self.assertTrue(inspect_receipt(legacy, expected_source='a' * 64)['full_suite_reported_passed'])

    def test_focused_pass_is_not_full_suite_pass(self):
        self.report.update(suite='checkpoint', tests_run=1)
        result = inspect_receipt(self.report, expected_source='a' * 64)
        self.assertFalse(result['full_suite_reported_passed'])
        self.assertEqual(result['reported_status'], 'passed')
        controls = {**self.report, 'suite': 'controls', 'tests_run': 5}
        self.assertFalse(inspect_receipt(controls, expected_source='a' * 64)['full_suite_reported_passed'])

    def test_contradictions_wrong_sources_and_missing_details_fail(self):
        for key, value in (('tests_run', 1), ('tests_run', True), ('errors', ['error']),
                           ('source_sha256', 'b' * 64), ('official_data_read', 0),
                           ('suite', 'other'), ('torch_version', ''), ('failures', None)):
            report = {**self.report, key: value}
            with self.assertRaises(AuditError):
                inspect_receipt(report, expected_source='a' * 64)

    def test_failed_and_blocked_reports_are_not_successes(self):
        failed = {**self.report, 'status': 'failed', 'errors': [{'test': 'checkpoint', 'detail': 'error'}]}
        self.assertFalse(inspect_receipt(failed, expected_source='a' * 64)['full_suite_reported_passed'])
        blocked = {**self.report, 'status': 'blocked', 'tests_run': 0, 'reason': 'missing runtime'}
        self.assertEqual(inspect_receipt(blocked, expected_source='a' * 64)['reported_status'], 'blocked')

    def test_git_revision_is_explicit_and_immutable(self):
        for value in ('HEAD', '../main', 'x' * 40, 'a' * 39):
            with self.assertRaises(AuditError):
                revision_identity(value)

    def test_cli_binds_original_bytes_and_never_overwrites_them(self):
        local = Path(__file__).resolve().parent.parent / '.local'
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            source = Path(temporary) / 'synthetic-report.json'
            output = Path(temporary) / 'receipt.json'
            raw = canonical(self.report)
            source.write_bytes(raw)
            args = ['--report', str(source), '--commit', 'b' * 40, '--output', str(output)]
            with patch('uavseg.runtime_receipt.revision_identity', return_value='a' * 64):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(args), 0)
                    self.assertEqual(main(args), 2)
                    self.assertEqual(main(args[:-1] + [str(source)]), 2)
            receipt = json.loads(output.read_bytes())
            self.assertEqual(receipt['report_file_sha256'], sha256(raw))
            self.assertEqual(source.read_bytes(), raw)
