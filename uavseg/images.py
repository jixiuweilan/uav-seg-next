"""Strict PNG decoding without conversion or label repair."""

from io import BytesIO

import numpy as np
from PIL import Image

from .common import AuditError, sha256

SIZE = (1024, 1024)
MAX_PNG_BYTES = 32 * 1024 * 1024


def inspect_png(data, name, *, mask=False):
    try:
        if len(data) > MAX_PNG_BYTES:
            raise AuditError("PNG exceeds the 32 MiB audit limit")
        with Image.open(BytesIO(data)) as im:
            if im.format != "PNG":
                raise AuditError("not a PNG")
            expected = "L" if mask else "RGB"
            if im.mode != expected or im.size != SIZE:
                raise AuditError(f"expected {expected} {SIZE}, got {im.mode} {im.size}")
            if im.n_frames != 1 or getattr(im, "is_animated", False):
                raise AuditError("animated PNG is not supported")
            if mask and "transparency" in im.info:
                raise AuditError("mask must not contain transparency")
            # Verify the actual PNG bit depth/color type, not a coerced decoder mode.
            if data[24:26] != bytes((8, 0 if mask else 2)):
                raise AuditError("expected an 8-bit grayscale/RGB PNG IHDR")
            im.verify()
        with Image.open(BytesIO(data)) as im:
            im.load()
            pixels = im.tobytes()
        record = {"bytes": len(data), "sha256": sha256(data),
                  "pixel_sha256": sha256(pixels), "mode": expected, "size": list(SIZE)}
        if mask:
            counts = np.bincount(np.frombuffer(pixels, dtype=np.uint8), minlength=256)
            invalid = np.flatnonzero(counts[9:]) + 9
            if invalid.size:
                raise AuditError(f"invalid label IDs: {invalid.tolist()}")
            record["class_counts"] = counts[:9].tolist()
        return record
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AuditError(f"{name}: {exc}") from exc


def inspect_file(path, *, mask=False):
    if path.stat().st_size > MAX_PNG_BYTES:
        raise AuditError(f"{path.name}: PNG exceeds the 32 MiB audit limit")
    return inspect_png(path.read_bytes(), path.name, mask=mask)
