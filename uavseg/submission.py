"""Validate and package the provisional flat submission layout."""

from pathlib import Path, PurePosixPath
import stat
import zipfile

from .common import AuditError, new_output, png_files, png_name
from .images import MAX_PNG_BYTES, inspect_png


def expected_names(document):
    rows = document["manifest"].get("test")
    if not isinstance(rows, list) or not rows:
        raise AuditError("audit must contain a nonempty test-image collection")
    names = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise AuditError("invalid test manifest row")
        name = png_name(row["id"] + ".png")
        image = row.get("image", {})
        if (not isinstance(image, dict) or image.get("mode") != "RGB"
                or image.get("size") != [1024, 1024]
                or not isinstance(image.get("path"), str)
                or PurePosixPath(image["path"]).name != name):
            raise AuditError(f"{name}: invalid test image metadata")
        if name in names:
            raise AuditError(f"duplicate expected filename: {name}")
        names.add(name)
    return names


def check_names(actual, expected):
    if actual != expected:
        raise AuditError(f"submission names: missing={sorted(expected-actual)[:10]}, "
                         f"extra={sorted(actual-expected)[:10]}")


def validate_zip(archive, document):
    expected = expected_names(document)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            names = [png_name(info.filename) for info in members]
            if len(names) != len(set(names)):
                raise AuditError("duplicate ZIP member names")
            check_names(set(names), expected)
            for info in members:
                file_type = stat.S_IFMT(info.external_attr >> 16)
                if file_type not in (0, stat.S_IFREG) or info.is_dir():
                    raise AuditError(f"{info.filename}: not a regular ZIP member")
                if info.file_size > MAX_PNG_BYTES:
                    raise AuditError(f"{info.filename}: PNG exceeds the 32 MiB audit limit")
                if info.flag_bits & 1:
                    raise AuditError(f"{info.filename}: encrypted member is not supported")
                inspect_png(zf.read(info), info.filename, mask=True)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise AuditError(f"invalid ZIP: {exc}") from exc
    return {"valid": True, "files": len(expected), "layout": "flat-provisional",
            "manifest_sha256": document["manifest_sha256"]}


def pack_submission(predictions, output, document, protected):
    predictions = Path(predictions)
    files = png_files(predictions)
    check_names(set(files), expected_names(document))
    with new_output(output, [*protected, predictions]) as temporary:
        # PNG is already compressed. STORE avoids zlib-dependent ZIP compression.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, path in files.items():
                if path.stat().st_size > MAX_PNG_BYTES:
                    raise AuditError(f"{name}: PNG exceeds the 32 MiB audit limit")
                data = path.read_bytes()
                inspect_png(data, name, mask=True)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_STORED
                zf.writestr(info, data)
        report = validate_zip(temporary, document)
        report["zip_sha256"] = file_sha256(temporary)
    return report


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
