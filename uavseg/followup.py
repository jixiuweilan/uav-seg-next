"""Portable local-original follow-up pages; JSON inputs only on the code terminal."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from .common import AuditError, canonical, new_output, read_json, relative_name, sha256
from .scenes import pair_id


def page_data(request):
    """Validate a versioned request and expose only the independent review inputs."""
    if not isinstance(request, dict):
        raise AuditError('跟进清单必须为对象')
    payload = {k: v for k, v in request.items() if k != 'request_sha256'}
    if request.get('request_sha256') != sha256(canonical(payload)):
        raise AuditError('跟进清单摘要不匹配')
    if request.get('schema_version') != 1 or request.get('kind') != 'scene-followup-request':
        raise AuditError('不是支持的跟进清单')
    for key in ('request_sha256', 'manifest_sha256', 'screen_sha256', 'source_file_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(request.get(key, ''))):
            raise AuditError('清单缺少有效身份摘要')
    images, names = {}, set()
    for row in request.get('required_images', []):
        key, image = row['id'], row['image']
        path = relative_name(image['path'])
        name = Path(path).name
        if not isinstance(key, str) or name != key + '.png' or key in images or name in names:
            raise AuditError('图像编号或文件名重复、不匹配')
        if (not re.fullmatch('[0-9a-f]{64}', image['sha256']) or
                image['size'] != [1024, 1024] or image['mode'] != 'RGB' or
                type(image['bytes']) is not int or image['bytes'] <= 0):
            raise AuditError('图像身份、尺寸或模式无效')
        images[key] = {k: image[k] for k in ('path', 'sha256', 'size', 'bytes')}
        images[key]['name'] = name
        names.add(name)
    pairs, seen, used = [], set(), set()
    for item in request.get('priority_relations', []):
        row = item.get('original_record') or item
        left, right, key = row['left'], row['right'], row['pair_id']
        if (left not in images or right not in images or left >= right or
                key != pair_id(left, right) or key in seen or item['priority'] not in (1, 2)):
            raise AuditError('关系编号、优先级或图像引用无效')
        # Do not expose previous judgments or reasons to the second reviewer.
        pairs.append({'left': left, 'right': right, 'pair_id': key, 'priority': item['priority']})
        seen.add(key)
        used.update((left, right))
    if not pairs or used != set(images):
        raise AuditError('清单为空或包含未使用图像')
    return {'base': {'schema_version': 1, 'kind': 'scene-followup-decisions',
                     **{k: request[k] for k in ('request_sha256', 'manifest_sha256',
                                               'screen_sha256', 'source_file_sha256')}},
            'images': images, 'pairs': pairs}


def validate_records(request, records):
    data = page_data(request)
    if not isinstance(records, dict) or any(records.get(k) != v for k, v in data['base'].items()):
        raise AuditError('复核记录类型或批次不匹配')
    rows = records.get('decisions')
    if not isinstance(rows, list):
        raise AuditError('复核记录必须包含列表')
    pairs = {p['pair_id']: p for p in data['pairs']}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError('无效复核条目')
        key = row.get('pair_id')
        if not isinstance(key, str) or key not in pairs or key in seen:
            raise AuditError('未知或重复关系')
        pair = pairs[key]
        if any(row.get(k) != pair[k] for k in ('left', 'right')):
            raise AuditError('关系的样本编号不匹配')
        if row.get('status') not in ('confirmed', 'rejected', 'uncertain'):
            raise AuditError('无效判断')
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('reason', 'reviewer')):
            raise AuditError('理由和复核人不能为空')
        stamp = row.get('reviewed_at', '')
        try:
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', stamp):
                raise ValueError()
            datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            raise AuditError('时间须为页面生成的含毫秒 UTC 时间') from None
        expected = {s: data['images'][pair[s]]['sha256'] for s in ('left', 'right')}
        if (row.get('image_sha256') != expected or row.get('viewed_originals') is not True or
                row.get('evidence') != 'full_resolution_images'):
            raise AuditError('原图身份或查看声明缺失')
        seen.add(key)
    return {'received': len(rows), 'pending': len(pairs) - len(rows),
            'image_evidence_independently_verified': False, 'decisions_merged': False}


def build_page(request_path, output):
    data = page_data(read_json(request_path))
    if '.local' not in Path(output).resolve().parts:
        raise AuditError('实际批次复核页面须保存在 .local/ 下')
    template = Path(__file__).with_name('followup.html').read_text(encoding='utf-8')
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    with new_output(output, [request_path]) as temporary:
        temporary.write_text(template.replace('/*PAGE_DATA*/', encoded), encoding='utf-8')
    return {'relations': len(data['pairs']), 'images': len(data['images']), 'raw_images_read': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description='生成本地原图复核页或校验回传 JSON，不读取原图')
    parser.add_argument('--request', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--records', type=Path)
    args = parser.parse_args(argv)
    try:
        result = (build_page(args.request, args.output) if args.output else
                  validate_records(read_json(args.request), read_json(args.records)))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (AuditError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'error': exc.strerror if isinstance(exc, OSError) else str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
