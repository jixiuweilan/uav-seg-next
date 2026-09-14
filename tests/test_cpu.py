"""Contract tests use generated pixels only; no competition data is bundled."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from PIL import Image

from uavseg.audit import audit_data, load_manifest, summary, validate_splits
from uavseg.common import AuditError, DataPaths, canonical, new_output, sha256, write_json
from uavseg.images import inspect_png
from uavseg.submission import pack_submission, validate_zip


def png(mode="L", value=0, size=(1024, 1024), **kwargs):
    stream = BytesIO()
    Image.new(mode, size, value).save(stream, format="PNG", **kwargs)
    return stream.getvalue()


class CPUContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "raw"
        for folder in ("train/images", "train/masks", "test/images"):
            (self.root / folder).mkdir(parents=True)
        for name, color in (("b", (12, 23, 34)), ("a", (1, 2, 3))):
            (self.root / f"train/images/{name}.png").write_bytes(png("RGB", color))
            (self.root / f"train/masks/{name}.png").write_bytes(png(value=1))
        (self.root / "test/images/t.png").write_bytes(png("RGB", (4, 5, 6)))
        self.config = self.base / "paths.json"
        self.settings = {"dataset_root": "raw", "train_images": "train/images",
                         "train_masks": "train/masks", "test_images": "test/images",
                         "access": "read-only"}
        self.config.write_bytes(canonical(self.settings))
        self.paths = DataPaths(self.config)
        self.splits = {}
        for key, ids in (("train", "a\n"), ("val", "b\n"), ("train_all", "a\nb\n")):
            path = self.base / f"{key}.txt"
            path.write_text(ids)
            self.splits[key] = path
        self.predictions = self.base / "predictions"
        self.predictions.mkdir()
        pixels = (np.arange(1024 * 1024) % 9).astype(np.uint8).reshape(1024, 1024)
        Image.fromarray(pixels).save(self.predictions / "t.png")
        # Minimal test expectation with the same shape as a real audited test row.
        manifest = {"schema_version": 1, "kind": "official-data-audit", "test": [
            {"id": "t", "image": {"mode": "RGB", "size": [1024, 1024],
                                     "path": "test/images/t.png"}}]}
        self.document = {"manifest": manifest, "manifest_sha256": sha256(canonical(manifest))}

    def audit(self):
        return audit_data(self.paths, self.splits)

    def archive(self, entries, name="candidate.zip"):
        target = self.base / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(target, "w") as zf:
                for entry, data in entries:
                    zf.writestr(entry, data)
        return target

    def test_manifest_determinism_counts_and_raw_preservation(self):
        before = {p: sha256(p.read_bytes()) for p in self.root.rglob("*.png")}
        first = self.audit()
        original = Path.iterdir

        def reversed_entries(path):
            return iter(reversed(list(original(path))))

        with patch.object(Path, "iterdir", reversed_entries):
            second = self.audit()
        self.assertEqual(canonical(first), canonical(second))
        self.assertEqual(before, {p: sha256(p.read_bytes()) for p in before})
        self.assertNotIn(str(self.root), canonical(first).decode())
        report = summary(first)
        self.assertEqual(report["train_pairs"], 2)
        self.assertEqual(report["test_images"], 1)
        self.assertEqual(report["class_counts"][1], 2 * 1024 * 1024)
        self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "train_all": 2})
        self.assertIn("same-scene leakage", report["not_checked"])

    def test_mask_change_changes_manifest_identity(self):
        before = self.audit()["manifest_sha256"]
        (self.root / "train/masks/a.png").write_bytes(png(value=8))
        self.assertNotEqual(before, self.audit()["manifest_sha256"])

    def test_pixel_digest_contract_uses_only_row_major_pixel_bytes(self):
        cases = (("RGB", (bytes((17, 83, 201)) * 512 + bytes((4, 5, 6)) * 512) * 1024),
                 ("L", (bytes((0, 8)) * 512) * 1024))
        for mode, pixels in cases:
            with self.subTest(mode=mode):
                stream = BytesIO()
                Image.frombytes(mode, (1024, 1024), pixels).save(stream, format="PNG")
                encoded = stream.getvalue()
                record = inspect_png(encoded, "fixture.png", mask=(mode == "L"))
                self.assertEqual(record["pixel_sha256"], hashlib.sha256(pixels).hexdigest())
                self.assertEqual(record["sha256"], hashlib.sha256(encoded).hexdigest())
                self.assertEqual(record["mode"], mode)
                self.assertEqual(record["size"], [1024, 1024])

    def test_pairing_error_names_sample(self):
        (self.root / "train/masks/a.png").unlink()
        with self.assertRaisesRegex(AuditError, "missing masks=.*a.png"):
            self.audit()

    def test_invalid_png_variants(self):
        variants = {"corrupt": b"invalid", "truncated": png()[:-20],
                    "size": png(size=(32, 32)), "palette": png("P"),
                    "rgb": png("RGB"), "rgba": png("RGBA"),
                    "16bit": png("I;16"), "id9": png(value=9),
                    "id255": png(value=255), "transparent": png(transparency=0)}
        stream = BytesIO()
        Image.new("L", (1024, 1024), 1).save(
            stream, format="PNG", save_all=True,
            append_images=[Image.new("L", (1024, 1024), 2)])
        variants["animated"] = stream.getvalue()
        for kind, data in variants.items():
            with self.subTest(kind=kind), self.assertRaisesRegex(AuditError, "sample.png"):
                inspect_png(data, "sample.png", mask=True)

    def test_audit_rejects_corrupt_and_invalid_masks_with_id(self):
        path = self.root / "train/masks/a.png"
        for data in (b"broken", png(value=9)):
            path.write_bytes(data)
            with self.assertRaisesRegex(AuditError, "a.png"):
                self.audit()

    def test_split_contracts(self):
        for key, value, message in (
                ("train", "a\na\n", "duplicate ID a"),
                ("val", "a\n", "intersection"),
                ("val", "x\n", "missing=.*b.*extra=.*x"),
                ("train_all", "a\n", "missing=.*b"),
                ("train", "\n", "blank"),
                ("train", "../a\n", "invalid relative")):
            path = self.splits[key]
            original = path.read_bytes()
            path.write_text(value)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(AuditError, message):
                validate_splits(["a", "b"], self.splits)
            path.write_bytes(original)
        with self.assertRaisesRegex(AuditError, "together"):
            validate_splits(["a", "b"], {"train": self.splits["train"]})

    def test_exact_pixel_duplicates_ignore_png_encoding(self):
        first = self.root / "train/images/a.png"
        second = self.root / "train/images/b.png"
        first.write_bytes(png("RGB", (1, 2, 3), compress_level=0))
        second.write_bytes(png("RGB", (1, 2, 3), compress_level=9))
        self.assertNotEqual(sha256(first.read_bytes()), sha256(second.read_bytes()))
        document = audit_data(self.paths)
        self.assertEqual(document["manifest"]["exact_duplicate_groups"], [["a", "b"]])
        with self.assertRaisesRegex(AuditError, "duplicates cross train/val"):
            self.audit()

    def test_config_and_symlink_boundaries(self):
        for key, value in (("access", "write"), ("train_images", "../outside"),
                           ("train_masks", "train/images")):
            config = {**self.settings, key: value}
            self.config.write_bytes(canonical(config))
            with self.subTest(key=key), self.assertRaises(AuditError):
                DataPaths(self.config)
        (self.root / "train/images/a.png").unlink()
        (self.root / "train/images/a.png").symlink_to(self.root / "train/images/b.png")
        with self.assertRaisesRegex(AuditError, "symlink"):
            self.audit()

    def test_outputs_cannot_write_raw_or_follow_alias_into_raw(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for output in (self.root / "new.json", alias / "new.json"):
            with self.subTest(output=output), self.assertRaisesRegex(AuditError, "protected"):
                write_json(output, {}, [self.root])
            self.assertFalse(output.exists())
        existing = self.base / "existing.json"
        existing.write_bytes(b"keep")
        with self.assertRaisesRegex(AuditError, "already exists"):
            write_json(existing, {}, [self.root])
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_failed_write_leaves_no_output_or_temporary(self):
        output = self.base / "incomplete.json"
        with self.assertRaisesRegex(AuditError, "intentional"):
            with new_output(output, [self.root]) as temporary:
                temporary.write_text("incomplete")
                raise AuditError("intentional failure")
        self.assertFalse(output.exists())
        self.assertEqual(list(self.base.glob(".uavseg-*")), [])

    def test_manifest_digest_tampering_is_rejected(self):
        output = self.base / "audit.json"
        write_json(output, self.document, [self.root])
        self.assertEqual(load_manifest(output), self.document)
        changed = json.loads(output.read_text())
        changed["manifest"]["test"][0]["id"] = "changed"
        output.write_bytes(canonical(changed))
        with self.assertRaisesRegex(AuditError, "digest mismatch"):
            load_manifest(output)

    def test_submission_round_trip_and_repeatable_bytes(self):
        before = (self.predictions / "t.png").read_bytes()
        outputs = [self.base / "one.zip", self.base / "two.zip"]
        for output in outputs:
            report = pack_submission(self.predictions, output, self.document, [self.root])
            self.assertTrue(report["valid"])
            self.assertEqual(report["files"], 1)
            with zipfile.ZipFile(output) as zf:
                self.assertEqual(zf.namelist(), ["t.png"])
                data = zf.read("t.png")
                self.assertEqual(data, before)
                with Image.open(BytesIO(data)) as decoded:
                    self.assertEqual(decoded.mode, "L")
                    self.assertEqual(set(np.asarray(decoded).ravel()), set(range(9)))
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(before, (self.predictions / "t.png").read_bytes())

    def test_zip_member_sets_and_paths(self):
        data = png()
        variants = [[], [("t.png", data), ("t.png", data)],
                    [("t.png", data), ("extra.png", data)], [("wrong.png", data)],
                    [("sub/t.png", data)], [("../t.png", data)],
                    [("/t.png", data)], [("sub\\t.png", data)], [("folder/", b"")]]
        for entries in variants:
            with self.subTest(names=[x[0] for x in entries]), self.assertRaises(AuditError):
                validate_zip(self.archive(entries), self.document)

    def test_pack_sorts_members_and_rejects_duplicate_manifest_ids(self):
        row = {"id": "s", "image": {"mode": "RGB", "size": [1024, 1024],
                                     "path": "test/images/s.png"}}
        self.document["manifest"]["test"].append(row)
        (self.predictions / "s.png").write_bytes(png(value=8))
        output = self.base / "sorted.zip"
        pack_submission(self.predictions, output, self.document, [self.root])
        with zipfile.ZipFile(output) as zf:
            self.assertEqual(zf.namelist(), ["s.png", "t.png"])
        self.document["manifest"]["test"].append(row)
        with self.assertRaisesRegex(AuditError, "duplicate expected"):
            validate_zip(output, self.document)

    def test_non_png_payload_and_crc_damage_are_rejected(self):
        stream = BytesIO()
        Image.new("L", (1024, 1024)).save(stream, format="BMP")
        damaged = bytearray(png(value=1))
        damaged[40] ^= 1
        for data in (stream.getvalue(), bytes(damaged)):
            with self.subTest(size=len(data)), self.assertRaises(AuditError):
                inspect_png(data, "sample.png", mask=True)

    def test_zip_rejects_bad_pngs_and_symlink(self):
        for data in (png("P"), png("RGB"), png(value=9), png(size=(8, 8)), b"broken"):
            with self.subTest(size=len(data)), self.assertRaisesRegex(AuditError, "t.png"):
                validate_zip(self.archive([("t.png", data)]), self.document)
        info = zipfile.ZipInfo("t.png")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(AuditError, "not a regular"):
            validate_zip(self.archive([(info, b"target")]), self.document)

    def test_zip_rejects_encryption_flag_before_reading_payload(self):
        archive = self.archive([("t.png", png())])
        validate_zip(archive, self.document)
        data = bytearray(archive.read_bytes())
        # ZIP writer clears flag_bits; set the flag in both actual headers.
        # This tests the encryption gate, not a particular encryption algorithm.
        central = data.index(b"PK\x01\x02")
        data[6] |= 1
        data[central + 8] |= 1
        archive.write_bytes(data)
        with patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('payload read')):
            with self.assertRaisesRegex(AuditError, 'encrypted member is not supported'):
                validate_zip(archive, self.document)

    def test_pack_invalid_inputs_is_atomic_and_protects_predictions(self):
        (self.predictions / "t.png").write_bytes(png(value=9))
        output = self.base / "bad.zip"
        with self.assertRaisesRegex(AuditError, "invalid label"):
            pack_submission(self.predictions, output, self.document, [self.root])
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(AuditError, "protected"):
            pack_submission(self.predictions, self.predictions / "new.zip",
                            self.document, [self.root])

    def test_cli_audit_pack_validate_and_failure_exit(self):
        output = self.base / "manifest.json"
        common = [sys.executable, "-m", "uavseg"]
        def run(*args):
            return subprocess.run([*common, *map(str, args)], capture_output=True, text=True)
        result = run("audit", "--config", self.config, "--output", output,
                     "--train-split", self.splits["train"], "--val-split", self.splits["val"],
                     "--all-split", self.splits["train_all"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["train_pairs"], 2)
        archive = self.base / "cli.zip"
        result = run("pack", "--config", self.config, "--manifest", output,
                     "--predictions", self.predictions, "--output", archive)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run("validate-zip", "--config", self.config, "--manifest", output,
                     "--archive", archive)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run("audit", "--config", self.config, "--output", self.root / "bad.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("protected", json.loads(result.stderr)["error"])
        self.assertFalse((self.root / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
