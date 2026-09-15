"""Serializer stream and publication failure contracts without PyTorch."""

from pathlib import Path
import tempfile
import unittest

from uavseg.common import AuditError, write_binary


class BinaryOutputTests(unittest.TestCase):
    def test_stream_is_open_during_write_and_closed_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'checkpoint.pt'
            streams = []
            def writer(stream):
                streams.append(stream)
                self.assertFalse(output.exists())
                self.assertFalse(stream.closed)
                self.assertTrue(Path(stream.name).name.startswith('.uavseg-'))
                stream.write(b'complete payload')
                stream.flush()
            write_binary(output, writer, [])
            self.assertTrue(streams[0].closed)
            self.assertEqual(output.read_bytes(), b'complete payload')
            self.assertEqual(list(Path(temporary).iterdir()), [output])

    def test_partial_write_failure_cleans_up_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'checkpoint.pt'
            streams = []
            def writer(stream):
                streams.append(stream)
                stream.write(b'partial')
                raise RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                write_binary(output, writer, [])
            self.assertTrue(streams[0].closed)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            write_binary(output, lambda stream: stream.write(b'complete'), [])
            self.assertEqual(output.read_bytes(), b'complete')

    def test_protected_and_existing_outputs_never_invoke_serializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'checkpoint.pt'
            def must_not_run(stream):
                self.fail('serializer was invoked for a protected destination')
            with self.assertRaises(AuditError):
                write_binary(output, must_not_run, [root])
            output.write_bytes(b'original')
            with self.assertRaises(AuditError):
                write_binary(output, must_not_run, [])
            self.assertEqual(output.read_bytes(), b'original')
