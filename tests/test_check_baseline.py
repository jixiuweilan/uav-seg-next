"""Missing-runtime reporting must not count skipped checks as success."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uavseg.check_baseline import main, source_identity


class RuntimeEntryTests(unittest.TestCase):
    def test_missing_torch_is_blocked_and_report_cannot_be_overwritten(self):
        local = Path(__file__).resolve().parent.parent / '.local'
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            output = Path(temporary) / 'report.json'
            with patch('uavseg.check_baseline.importlib.util.find_spec', return_value=None):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['--output', str(output)]), 2)
                    before = output.read_bytes()
                    self.assertEqual(main(['--output', str(output)]), 2)
                    self.assertEqual(output.read_bytes(), before)
            result = json.loads(before)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['tests_run'], 0)
            self.assertFalse(result['official_data_read'] or result['formal_training_started'])
            self.assertEqual(result['source_sha256'], source_identity())

    def test_output_outside_local_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report.json'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--output', str(output)]), 2)
            self.assertFalse(output.exists())
