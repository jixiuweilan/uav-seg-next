"""Deterministic official-data inventory and reference split validation."""

from collections import defaultdict
from pathlib import Path
import platform

import numpy as np
from PIL import __version__ as pillow_version

from . import __version__
from .common import AuditError, canonical, png_files, png_name, read_json, sha256
from .images import inspect_file

UNCHECKED = ["near duplicates", "same-scene leakage", "annotation correctness",
             "source flight provenance", "train-test overlap"]


def read_ids(path):
    raw = Path(path).read_bytes()
    ids = raw.decode("utf-8").splitlines()
    if not ids:
        raise AuditError(f"{Path(path).name}: empty split")
    seen = set()
    for value in ids:
        if value != value.strip() or not value:
            raise AuditError(f"{Path(path).name}: blank/whitespace split ID")
        png_name(value + ".png")
        if value in seen:
            raise AuditError(f"{Path(path).name}: duplicate ID {value}")
        seen.add(value)
    return {"ids": sorted(ids), "source_sha256": sha256(raw)}


def validate_splits(train_ids, paths):
    if not paths:
        return None
    if set(paths) != {"train", "val", "train_all"}:
        raise AuditError("provide train, val and train_all split files together")
    result = {key: read_ids(path) for key, path in paths.items()}
    train, val, all_ids = (set(result[k]["ids"]) for k in ("train", "val", "train_all"))
    if train & val:
        raise AuditError(f"train/val intersection: {sorted(train & val)[:10]}")
    expected = set(train_ids)
    for name, actual in (("train+val", train | val), ("train_all", all_ids)):
        if actual != expected:
            raise AuditError(f"{name}: missing={sorted(expected-actual)[:10]}, "
                             f"extra={sorted(actual-expected)[:10]}")
    return result


def audit_data(paths, split_paths=None, progress=None):
    files = {key: png_files(folder) for key, folder in paths.folders.items()}
    images, masks = files["train_images"], files["train_masks"]
    if images.keys() != masks.keys():
        raise AuditError(f"pair mismatch: missing masks={sorted(images.keys()-masks.keys())[:10]}, "
                         f"missing images={sorted(masks.keys()-images.keys())[:10]}")
    splits = validate_splits([name[:-4] for name in images], split_paths)
    manifest = {"schema_version": 1, "kind": "official-data-audit", "train": [],
                "test": [], "splits": splits, "not_checked": UNCHECKED.copy()}
    if splits is None:
        manifest["not_checked"].append("split membership")
    duplicates = defaultdict(list)
    for index, (name, path) in enumerate(images.items(), 1):
        image = inspect_file(path)
        mask = inspect_file(masks[name], mask=True)
        image["path"] = path.relative_to(paths.root).as_posix()
        mask["path"] = masks[name].relative_to(paths.root).as_posix()
        sample_id = name[:-4]
        manifest["train"].append({"id": sample_id, "image": image, "mask": mask})
        duplicates[image["pixel_sha256"]].append(sample_id)
        if progress and (index % 250 == 0 or index == len(images)):
            progress(f"decoded training pairs: {index}/{len(images)}")
    for name, path in files["test_images"].items():
        image = inspect_file(path)
        image["path"] = path.relative_to(paths.root).as_posix()
        manifest["test"].append({"id": name[:-4], "image": image})
    groups = sorted(sorted(ids) for ids in duplicates.values() if len(ids) > 1)
    manifest["exact_duplicate_groups"] = groups
    if splits:
        train = set(splits["train"]["ids"])
        val = set(splits["val"]["ids"])
        crossing = [ids for ids in groups if train.intersection(ids) and val.intersection(ids)]
        if crossing:
            raise AuditError(f"exact decoded duplicates cross train/val: {crossing[:10]}")
    package = Path(__file__).parent
    sources = {p.name: sha256(p.read_bytes()) for p in sorted(package.glob("*.py"))}
    return {"manifest": manifest, "manifest_sha256": sha256(canonical(manifest)),
            "producer": {"version": __version__, "python": platform.python_version(),
                         "pillow": pillow_version, "numpy": np.__version__,
                         "source_sha256": sha256(canonical(sources))}}


def load_manifest(path):
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("manifest"), dict):
        raise AuditError("invalid audit document")
    manifest = document["manifest"]
    if sha256(canonical(manifest)) != document.get("manifest_sha256"):
        raise AuditError("audit manifest digest mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "official-data-audit":
        raise AuditError("unsupported audit schema")
    return document


def summary(document):
    manifest = document["manifest"]
    counts = [sum(row["mask"]["class_counts"][k] for row in manifest["train"])
              for k in range(9)]
    return {"manifest_sha256": document["manifest_sha256"],
            "train_pairs": len(manifest["train"]), "test_images": len(manifest["test"]),
            "class_counts": counts, "exact_duplicate_groups": manifest["exact_duplicate_groups"],
            "split_counts": {key: len(value["ids"]) for key, value in
                             (manifest["splits"] or {}).items()},
            "not_checked": manifest["not_checked"], "producer": document["producer"]}
