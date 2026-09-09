"""Training-image similarity cues, with no automatic scene decisions."""

from dataclasses import dataclass
from io import BytesIO
import heapq
from pathlib import Path
import platform
import random

import numpy as np
from PIL import Image, __version__ as pillow_version

from . import __version__
from .common import AuditError, canonical, png_files, png_name, sha256

POSES = ("r0", "r90", "r180", "r270", "flip-r0", "flip-r90", "flip-r180", "flip-r270")
DEFAULTS = {"hash_max": 8, "rms_max": 0.45, "min_std": 5.0,
            "max_per_partition": 100, "controls": 20, "seed": 0}


def producer():
    sources = {p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"version": __version__, "python": platform.python_version(),
            "pillow": pillow_version, "numpy": np.__version__,
            "source_sha256": sha256(canonical(sources))}


def training_rows(document):
    rows = document["manifest"].get("train")
    if not isinstance(rows, list) or len(rows) < 2:
        raise AuditError("scene screening requires at least two audited training images")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise AuditError("invalid training manifest row")
        name = png_name(row["id"] + ".png")
        image = row.get("image", {})
        if (not isinstance(image, dict) or image.get("mode") != "RGB"
                or image.get("size") != [1024, 1024]
                or not isinstance(image.get("path"), str)
                or Path(image["path"]).name != name):
            raise AuditError(f"{name}: invalid training-image metadata")
        if row["id"] in result:
            raise AuditError(f"duplicate training ID: {row['id']}")
        result[row["id"]] = row
    return dict(sorted(result.items()))


def split_membership(document, ids):
    splits = document["manifest"].get("splits")
    if splits is None:
        return {}
    try:
        train, val, all_ids = (splits[k]["ids"] for k in ("train", "val", "train_all"))
        for values in (train, val, all_ids):
            if len(values) != len(set(values)):
                raise AuditError("duplicate ID in audited split")
        if set(train) & set(val) or set(train) | set(val) != set(ids) or set(all_ids) != set(ids):
            raise AuditError("audited split coverage/intersection is invalid")
        return {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val")}
    except (KeyError, TypeError) as exc:
        raise AuditError("invalid audited splits") from exc


def verified_image(paths, row):
    path = paths.folders["train_images"] / png_name(row["id"] + ".png")
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{row['id']}: expected a regular training image")
    if path.relative_to(paths.root).as_posix() != row["image"]["path"]:
        raise AuditError(f"{row['id']}: training path differs from audited manifest")
    data = path.read_bytes()
    if sha256(data) != row["image"]["sha256"]:
        raise AuditError(f"{row['id']}: image changed since audit")
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise AuditError(f"{row['id']}: invalid audited image mode/size")
        return image.copy()


def transformed(array, pose):
    """Mirror left/right first, then rotate counterclockwise in quarter turns."""
    return np.rot90(np.fliplr(array) if pose >= 4 else array, pose % 4)


def average_hash(gray):
    cells = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    bits = np.packbits((cells > cells.mean()).ravel(), bitorder="big")
    return np.uint64(int.from_bytes(bits.tobytes(), "big"))


@dataclass
class Features:
    ids: list
    hashes: np.ndarray
    normalized: np.ndarray
    stds: np.ndarray
    pixel_hashes: list


def extract_features(paths, document, progress=None):
    rows = training_rows(document)
    if set(png_files(paths.folders["train_images"])) != {s + ".png" for s in rows}:
        raise AuditError("training-image collection differs from audited manifest")
    hashes, normalized, stds = [], [], []
    for index, row in enumerate(rows.values(), 1):
        with verified_image(paths, row) as image:
            gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
        values = gray.astype(np.float32)
        std = float(values.std())
        hashes.append([average_hash(transformed(gray, pose)) for pose in range(8)])
        normalized.append((values - values.mean()) / max(std, 1.0))
        stds.append(std)
        if progress and (index % 250 == 0 or index == len(rows)):
            progress(f"verified training-image descriptors: {index}/{len(rows)}")
    return Features(list(rows), np.asarray(hashes, dtype=np.uint64),
                    np.asarray(normalized, dtype=np.float32), np.asarray(stds),
                    [row["image"]["pixel_sha256"] for row in rows.values()])


def parameters(overrides=None):
    result = {**DEFAULTS, **(overrides or {})}
    if set(result) != set(DEFAULTS):
        raise AuditError("unknown scene-screening parameter")
    for key, minimum, maximum in (("hash_max", 0, 64), ("max_per_partition", 1, 10000),
                                   ("controls", 0, 10000), ("seed", 0, 2**32 - 1)):
        if type(result[key]) is not int or not minimum <= result[key] <= maximum:
            raise AuditError(f"invalid screening parameter: {key}")
    for key in ("rms_max", "min_std"):
        if not isinstance(result[key], (int, float)) or not np.isfinite(result[key]) or result[key] < 0:
            raise AuditError(f"invalid screening parameter: {key}")
    return result


def pair_id(left, right):
    if left >= right:
        raise AuditError("pair IDs must be distinct and sorted")
    return sha256(canonical([left, right]))


def metrics(features, i, j, poses=range(8)):
    values = []
    for pose in poses:
        delta = features.normalized[i] - transformed(features.normalized[j], int(pose))
        rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
        hamming = int(np.bitwise_count(features.hashes[i, 0] ^ features.hashes[j, pose]))
        values.append((rms, hamming, int(pose)))
    return min(values)


def qualifying_score(features, i, j, config):
    if features.pixel_hashes[i] == features.pixel_hashes[j]:
        return (0.0, 0, 0)
    if min(features.stds[i], features.stds[j]) < config["min_std"]:
        return None
    hamming = np.bitwise_count(features.hashes[i, 0] ^ features.hashes[j])
    poses = np.flatnonzero(hamming <= config["hash_max"])
    if not len(poses):
        return None
    score = metrics(features, i, j, poses)
    return score if score[0] <= config["rms_max"] else None


def screen_features(features, document, config=None, progress=None):
    config = parameters(config)
    ids = features.ids
    if ids != list(training_rows(document)):
        raise AuditError("descriptor IDs differ from training manifest")
    membership = split_membership(document, ids)
    heaps = {key: [] for key in ("cross_split", "within_split", "unassigned")}
    matched = dict.fromkeys(heaps, 0)

    def row_for(i, j, score, queue):
        left, right = ids[i], ids[j]
        cross = membership[left] != membership[right] if membership else None
        return {"pair_id": pair_id(left, right), "left": left, "right": right,
                "queue": queue, "cross_split": cross, "rms": round(score[0], 8),
                "hamming": score[1], "pose": POSES[score[2]],
                "exact_pixels": features.pixel_hashes[i] == features.pixel_hashes[j],
                "low_texture": bool(min(features.stds[i], features.stds[j]) < config["min_std"])}

    for i in range(len(ids) - 1):
        distances = np.bitwise_count(features.hashes[i, 0] ^ features.hashes[i + 1:])
        for j in np.flatnonzero(np.any(distances <= config["hash_max"], axis=1)) + i + 1:
            j = int(j)
            score = qualifying_score(features, i, j, config)
            if score is None:
                continue
            row = row_for(i, j, score, "candidate")
            key = "unassigned" if row["cross_split"] is None else (
                "cross_split" if row["cross_split"] else "within_split")
            matched[key] += 1
            # A bounded heap retains strongest cues; overflow is explicitly counted.
            entry = (-int(not row["exact_pixels"]), -score[0], -score[1], -i, -j, row)
            if len(heaps[key]) < config["max_per_partition"]:
                heapq.heappush(heaps[key], entry)
            elif entry > heaps[key][0]:
                heapq.heapreplace(heaps[key], entry)
        if progress and (i % 500 == 0 or i == len(ids) - 2):
            progress(f"pair-screen rows: {i + 1}/{len(ids) - 1}")
    rows = [entry[-1] for heap in heaps.values() for entry in heap]
    total = len(ids) * (len(ids) - 1) // 2
    target = min(config["controls"], total - sum(matched.values()))
    rng = random.Random(config["seed"])
    controls = set()
    for _ in range(max(1000, target * 100)):
        if len(controls) == target:
            break
        i, j = sorted(rng.sample(range(len(ids)), 2))
        if (i, j) in controls or qualifying_score(features, i, j, config) is not None:
            continue
        controls.add((i, j))
        rows.append(row_for(i, j, metrics(features, i, j), "control"))
    rows.sort(key=lambda r: (r["queue"] != "candidate", r["cross_split"] is not True,
                              not r["exact_pixels"], r["rms"], r["hamming"], r["left"], r["right"]))
    payload = {"schema_version": 1, "kind": "scene-candidates",
               "manifest_sha256": document["manifest_sha256"], "train_images": len(ids),
               "method": {"name": "gray-ahash64-normalized-rms-v1", "thumbnail_size": 32,
                          "resize": "Pillow L/LANCZOS", "poses": list(POSES), **config},
               "total_pairs": total, "matched_by_partition": matched,
               "retained_by_partition": {key: len(heap) for key, heap in heaps.items()},
               "omitted_by_partition": {key: matched[key] - len(heap) for key, heap in heaps.items()},
               "controls_retained": len(controls), "controls_target": config["controls"],
               "low_texture_ids": [s for s, std in zip(ids, features.stds) if std < config["min_std"]],
               "scene_independence_certified": False, "pairs": rows}
    return {"screen": payload, "screen_sha256": sha256(canonical(payload)), "producer": producer()}


def validate_screen(screen, document):
    payload = screen.get("screen") if isinstance(screen, dict) else None
    if not isinstance(payload, dict) or sha256(canonical(payload)) != screen.get("screen_sha256"):
        raise AuditError("candidate document digest mismatch")
    if payload.get("schema_version") != 1 or payload.get("kind") != "scene-candidates":
        raise AuditError("unsupported candidate schema")
    if payload.get("manifest_sha256") != document["manifest_sha256"]:
        raise AuditError("candidate document belongs to a different manifest")
    ids = training_rows(document)
    membership = split_membership(document, ids)
    if not isinstance(payload.get("pairs"), list):
        raise AuditError("candidate pairs must be a list")
    for field in ("matched_by_partition", "retained_by_partition", "omitted_by_partition"):
        counts = payload.get(field)
        if (not isinstance(counts, dict) or set(counts) != {"cross_split", "within_split", "unassigned"}
                or any(type(v) is not int or v < 0 for v in counts.values())):
            raise AuditError(f"invalid candidate counts: {field}")
    pairs = {}
    for row in payload.get("pairs", []):
        if not isinstance(row, dict) or row.get("left") not in ids or row.get("right") not in ids:
            raise AuditError("candidate references an unknown training ID")
        expected = pair_id(row["left"], row["right"])
        if row.get("pair_id") != expected or expected in pairs:
            raise AuditError("invalid or duplicate candidate pair ID")
        if row.get("queue") not in ("candidate", "control") or row.get("pose") not in POSES:
            raise AuditError("invalid candidate queue or pose")
        if (type(row.get("rms")) not in (float, int) or not np.isfinite(row["rms"]) or row["rms"] < 0
                or type(row.get("hamming")) is not int or not 0 <= row["hamming"] <= 64
                or type(row.get("exact_pixels")) is not bool or type(row.get("low_texture")) is not bool):
            raise AuditError("invalid candidate score or flags")
        cross = membership[row["left"]] != membership[row["right"]] if membership else None
        if row.get("cross_split") != cross:
            raise AuditError("candidate split annotation differs from manifest")
        pairs[expected] = row
    return pairs
