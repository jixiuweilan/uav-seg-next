"""Draft group-respecting splits from audited metadata; never load raw images."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from .audit import load_manifest
from .common import AuditError, canonical, read_json, sha256, write_json
from .review import group_report
from .scenes import split_membership, training_rows


def propose_split(document, screen, decisions, *, seed=0, validation_count=None):
    if type(seed) is not int:
        raise AuditError('划分种子须为整数')
    if sha256(canonical(document['manifest'])) != document.get('manifest_sha256'):
        raise AuditError('审计清单摘要不匹配')
    rows = training_rows(document)
    reference = split_membership(document, rows)
    if not reference:
        raise AuditError('候选划分需要完整的参考划分')
    target = sum(v == 'val' for v in reference.values()) if validation_count is None else validation_count
    if type(target) is not int or not 0 < target < len(rows):
        raise AuditError('验证样本目标须为1至总数减1的整数')
    report = group_report(document, screen, decisions)['report']
    if report['contradictory_rejections']:
        raise AuditError('确认关系存在组内排除矛盾，不能生成候选划分')
    parent = {key: key for key in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(members):
        roots = sorted({root(key) for key in members})
        for key in roots[1:]:
            parent[key] = roots[0]

    for group in report['groups']:
        union(group['members'])
    exact = {}
    for key, row in rows.items():
        digest = row['image'].get('pixel_sha256', '')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise AuditError('审计清单缺少有效像素摘要')
        exact.setdefault(digest, []).append(key)
    for members in exact.values():
        union(members)
    for record in decisions['decisions']:
        if record['status'] == 'rejected' and root(record['left']) == root(record['right']):
            raise AuditError('排除关系与已知像素重复或确认连接矛盾')
    components = {}
    for key in rows:
        components.setdefault(root(key), []).append(key)
    components = sorted(components.values())

    def rank(kind, members):
        return sha256(canonical([seed, kind, members]))

    allocation = {}
    for members in components:
        counts = Counter(reference[key] for key in members)
        if counts['train'] == counts['val']:
            side = 'val' if int(rank('tie', members), 16) % 2 else 'train'
        else:
            side = 'train' if counts['train'] > counts['val'] else 'val'
        allocation.update(dict.fromkeys(members, side))
    # Preserve every multi-image component. Only ungrouped singletons compensate
    # for the validation-count change; never search alternate seeds for scores.
    current = sum(v == 'val' for v in allocation.values())
    source, destination = ('train', 'val') if current < target else ('val', 'train')
    singles = sorted((members for members in components if len(members) == 1 and
                      allocation[members[0]] == source), key=lambda ids: rank('rebalance', ids))
    for members in singles[:abs(target - current)]:
        allocation[members[0]] = destination
    train = sorted(key for key, side in allocation.items() if side == 'train')
    val = sorted(key for key, side in allocation.items() if side == 'val')
    if not train or not val:
        raise AuditError('现有整组约束不能形成非空训练与验证候选集')
    coverage = {}
    for side, ids in (('train', train), ('val', val)):
        pixels, samples = [0] * 9, [0] * 9
        for key in ids:
            mask = rows[key].get('mask', {})
            counts = mask.get('class_counts')
            if (not isinstance(counts, list) or len(counts) != 9 or
                    any(type(n) is not int or n < 0 for n in counts) or
                    sum(counts) != 1024 * 1024):
                raise AuditError('审计标签类别统计无效')
            for k, n in enumerate(counts):
                pixels[k] += n
                samples[k] += int(n > 0)
        coverage[side] = {'pixels_by_class': pixels, 'samples_by_class': samples,
                          'missing_evaluated_classes': [k for k in range(1, 9) if not pixels[k]]}
    constraints = [{'group_id': sha256(canonical([document['manifest_sha256'], members])),
                    'members': members, 'split': allocation[members[0]]}
                   for members in components if len(members) > 1]
    assert not set(train) & set(val) and set(train) | set(val) == set(rows)
    assert all(len({allocation[key] for key in members}) == 1 for members in components)
    result = {'schema_version': 1, 'kind': 'candidate-group-split', 'status': 'draft',
              'policy': 'preserve_component_majority_then_seeded_singletons_v1', 'seed': seed,
              'manifest_sha256': document['manifest_sha256'], 'screen_sha256': screen['screen_sha256'],
              'decisions_sha256': report['decisions_sha256'], 'validation_target': target,
              'validation_target_delta': len(val) - target, 'train_ids': train, 'val_ids': val,
              'groups': constraints, 'class_coverage': coverage,
              'moved_from_reference': [{'id': key, 'from': reference[key], 'to': allocation[key]}
                                       for key in rows if reference[key] != allocation[key]],
              'unreviewed_relations': len(report['unreviewed_pair_ids']),
              'uncertain_relations': len(report['uncertain_pair_ids']),
              'omitted_candidates': report['omitted_candidates'],
              'declared_groups_over_20_members': len(report['groups_over_20_members']),
              'known_constraint_crossings': 0, 'scene_independence_certified': False,
              'frozen': False, 'training_authorized': False, 'raw_files_read': 0}
    return {'split': result, 'split_sha256': sha256(canonical(result))}


def main(argv=None):
    parser = argparse.ArgumentParser(description='从已审计 JSON 生成整组候选划分，不读取原图或修改参考划分')
    for name in ('manifest', 'screen', 'decisions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-count', type=int)
    args = parser.parse_args(argv)
    try:
        if '.local' not in args.output.resolve().parts:
            raise AuditError('实际批次候选划分须保存在 .local/ 下')
        result = propose_split(load_manifest(args.manifest), read_json(args.screen),
                               read_json(args.decisions), seed=args.seed,
                               validation_count=args.validation_count)
        write_json(args.output, result, [args.manifest, args.screen, args.decisions])
        split = result['split']
        print(json.dumps({'train': len(split['train_ids']), 'val': len(split['val_ids']),
                          'moved': len(split['moved_from_reference']), 'status': split['status'],
                          'split_sha256': result['split_sha256']}, ensure_ascii=False))
        return 0
    except (AuditError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'error': exc.strerror if isinstance(exc, OSError) else str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
