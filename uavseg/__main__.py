"""Run with python -m uavseg; every command is CPU-only."""

import argparse
import json
from pathlib import Path
import sys

from .audit import audit_data, load_manifest, summary
from .common import AuditError, DataPaths, canonical, check_output, read_json, write_json
from .review import group_report, review_page
from .scenes import DEFAULTS, extract_features, screen_features
from .submission import pack_submission, validate_zip


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="decode official data and validate reference splits")
    audit.add_argument("--config", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    for key in ("train", "val", "all"):
        audit.add_argument(f"--{key}-split", type=Path)
    for name in ("pack", "validate-zip"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--manifest", required=True, type=Path)
        if name == "pack":
            child.add_argument("--predictions", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        else:
            child.add_argument("--archive", required=True, type=Path)
    for name in ("screen-scenes", "review-scenes", "check-groups"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        if name == "screen-scenes":
            for key, value in DEFAULTS.items():
                child.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
        else:
            child.add_argument("--screen", required=True, type=Path)
            if name == "check-groups":
                child.add_argument("--decisions", type=Path)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        paths = DataPaths(args.config)
        protected = [paths.root, paths.config]
        if args.command == "audit":
            splits = {key: path for key, path in
                      (("train", args.train_split), ("val", args.val_split),
                       ("train_all", args.all_split)) if path is not None}
            protected.extend(splits.values())
            check_output(args.output, protected)
            document = audit_data(paths, splits, progress=lambda s: print(s, file=sys.stderr))
            write_json(args.output, document, protected)
            report = summary(document)
        elif args.command in ("screen-scenes", "review-scenes", "check-groups"):
            document = load_manifest(args.manifest)
            protected.append(args.manifest)
            if hasattr(args, "screen"):
                protected.append(args.screen)
            if getattr(args, "decisions", None):
                protected.append(args.decisions)
            check_output(args.output, protected)
            if args.command == "screen-scenes":
                progress = lambda s: print(s, file=sys.stderr, flush=True)
                features = extract_features(paths, document, progress)
                result = screen_features(features, document, {k: getattr(args, k) for k in DEFAULTS}, progress)
                write_json(args.output, result, protected)
                report = {k: v for k, v in result["screen"].items() if k not in ("pairs", "low_texture_ids")}
                report["low_texture_images"] = len(result["screen"]["low_texture_ids"])
                report["screen_sha256"] = result["screen_sha256"]
            elif args.command == "review-scenes":
                report = review_page(paths, document, read_json(args.screen), args.output, protected)
            else:
                result = group_report(document, read_json(args.screen),
                                      read_json(args.decisions) if args.decisions else None)
                write_json(args.output, result, protected)
                report = {k: v for k, v in result["report"].items()
                          if k not in ("groups", "ungrouped_ids", "unreviewed_pair_ids", "uncertain_pair_ids")}
                report["groups"] = len(result["report"]["groups"])
                print(canonical(report).decode("utf-8"), end="")
                return 2 if report["status"] in ("known_split_conflict", "inconsistent_review") else 0
        else:
            document = load_manifest(args.manifest)
            protected.append(args.manifest)
            if args.command == "pack":
                report = pack_submission(args.predictions, args.output, document, protected)
            else:
                report = validate_zip(args.archive, document)
        print(canonical(report).decode("utf-8"), end="")
        return 0
    except (AuditError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        # OS errors can contain machine paths; report the operation without the path.
        message = exc.strerror if isinstance(exc, OSError) else str(exc)
        print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
