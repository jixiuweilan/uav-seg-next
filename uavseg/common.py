"""Portable identities and read-only input/output boundaries."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


class AuditError(ValueError):
    """An input does not satisfy the documented contract."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise AuditError("expected a nonempty portable relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in (".", "..", "") for x in value.split("/")):
        raise AuditError(f"invalid relative path: {value!r}")
    if ":" in value or any(ord(c) < 32 for c in value):
        raise AuditError(f"invalid relative path: {value!r}")
    return value


def png_name(value):
    relative_name(value)
    if "/" in value or not value.endswith(".png") or value == ".png":
        raise AuditError(f"expected a flat PNG filename: {value!r}")
    return value


class DataPaths:
    def __init__(self, config):
        self.config = Path(config).resolve()
        value = read_json(self.config)
        if not isinstance(value, dict) or value.get("access") != "read-only":
            raise AuditError("data configuration must declare access=read-only")
        root = value.get("dataset_root")
        if not isinstance(root, str) or not root:
            raise AuditError("dataset_root must be a nonempty path")
        root = Path(root).expanduser()
        self.root = (root if root.is_absolute() else self.config.parent / root).resolve()
        if not self.root.is_dir():
            raise AuditError("dataset_root is not an accessible directory")
        self.folders = {}
        for key in ("train_images", "train_masks", "test_images"):
            name = relative_name(value.get(key))
            path = self.root / name
            if not path.resolve().is_relative_to(self.root) or not path.is_dir():
                raise AuditError(f"{key}: directory missing or outside dataset_root")
            self.folders[key] = path
        resolved = [p.resolve() for p in self.folders.values()]
        if len(set(resolved)) != 3:
            raise AuditError("image/mask/test directories must be distinct")


def png_files(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.iterdir()):
        png_name(path.name)
        if path.is_symlink() or not path.is_file():
            raise AuditError(f"{path.name}: expected a regular PNG file, no symlink")
        files[path.name] = path
    if not files:
        raise AuditError(f"{directory.name}: empty PNG collection")
    return files


def check_output(output, protected):
    output = Path(output)
    resolved = output.resolve()
    for source in protected:
        source = Path(source).resolve()
        if resolved == source or (source.is_dir() and resolved.is_relative_to(source)):
            raise AuditError("output is inside a protected input location")
    if output.exists() or output.is_symlink():
        raise AuditError("output already exists; choose a new versioned output")
    return output


@contextmanager
def new_output(output, protected):
    """Publish only complete artifacts, never replace an existing output."""
    output = check_output(output, protected)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uavseg-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        yield temporary
        check_output(output, protected)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(output, value, protected):
    with new_output(output, protected) as temporary:
        temporary.write_bytes(canonical(value))
