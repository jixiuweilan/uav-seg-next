"""Read audited bytes on the execution machine; tests use synthetic PNGs only."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .common import AuditError, canonical, png_name, relative_name, sha256
from .data import prepare_pair
from .images import MAX_PNG_BYTES, inspect_png


def validate_document(document):
    if not isinstance(document, dict) or not isinstance(document.get('manifest'), dict):
        raise AuditError('无效审计清单')
    manifest = document['manifest']
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != 'official-data-audit' or
            sha256(canonical(manifest)) != document.get('manifest_sha256')):
        raise AuditError('审计清单版本或摘要不匹配')


def indexed_rows(document, collection):
    validate_document(document)
    rows = document['manifest'].get(collection)
    if not isinstance(rows, list) or not rows:
        raise AuditError('审计清单缺少样本')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            raise AuditError('样本记录无效')
        name = png_name(row['id'] + '.png')
        if row['id'] in result:
            raise AuditError('样本 ID 重复')
        for field in ('image', 'mask') if collection == 'train' else ('image',):
            record = row.get(field)
            if (not isinstance(record, dict) or
                    Path(relative_name(record.get('path'))).name != name):
                raise AuditError('样本文件名与 ID 不一致')
        result[row['id']] = row
    return result


def candidate_ids(document, proposal, partition):
    rows = indexed_rows(document, 'train')
    if partition not in ('train', 'val') or not isinstance(proposal, dict):
        raise AuditError('划分只能是 train 或 val')
    split = proposal.get('split')
    if (not isinstance(split, dict) or split.get('kind') != 'candidate-group-split' or
            split.get('schema_version') != 1 or
            sha256(canonical(split)) != proposal.get('split_sha256') or
            split.get('manifest_sha256') != document['manifest_sha256']):
        raise AuditError('候选划分摘要或审计来源不匹配')
    parts = []
    for key in ('train_ids', 'val_ids'):
        ids = split.get(key)
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids) or
                ids != sorted(set(ids))):
            raise AuditError('候选划分 ID 须非空、有序且无重复')
        parts.append(set(ids))
    if parts[0] & parts[1] or parts[0] | parts[1] != set(rows):
        raise AuditError('候选划分不满足互斥和完整覆盖')
    groups = split.get('groups')
    if not isinstance(groups, list):
        raise AuditError('缺少整组约束')
    for group in groups:
        members = group.get('members') if isinstance(group, dict) else None
        if (not isinstance(members, list) or len(members) < 2 or
                any(not isinstance(x, str) for x in members) or
                len(set(members)) != len(members) or not set(members) <= set(rows) or
                group.get('split') not in ('train', 'val')):
            raise AuditError('整组约束无效')
        side = 0 if group['split'] == 'train' else 1
        if not set(members) <= parts[side]:
            raise AuditError('候选划分破坏已记录整组约束')
    return tuple(split[partition + '_ids'])


class AuditedSamples:
    def __init__(self, root, document, *, partition, proposal=None):
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise AuditError('数据根目录不可用')
        if partition not in ('train', 'val', 'test'):
            raise AuditError('未知样本划分')
        # Freeze a private snapshot so caller mutations cannot change later reads.
        document = deepcopy(document)
        self._rows = indexed_rows(document, 'test' if partition == 'test' else 'train')
        self.ids = (tuple(sorted(self._rows)) if partition == 'test' else
                    candidate_ids(document, proposal, partition))
        self.partition = partition
        self.manifest_sha256 = document['manifest_sha256']
        self.split_sha256 = None if partition == 'test' else proposal['split_sha256']

    def _read(self, record, *, mask=False):
        name = relative_name(record.get('path'))
        path = self.root
        for part in Path(name).parts:
            path = path / part
            if path.is_symlink():
                raise AuditError('不接受符号链接数据路径')
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise AuditError('数据文件缺失或越过根目录')
        with path.open('rb') as stream:
            data = stream.read(MAX_PNG_BYTES + 1)
        if sha256(data) != record.get('sha256'):
            raise AuditError(f'{name}: 文件身份与审计记录不匹配')
        actual = inspect_png(data, name, mask=mask)
        if any(record.get(key) != value for key, value in actual.items()):
            raise AuditError(f'{name}: 解码统计与审计记录不匹配')
        with Image.open(BytesIO(data)) as image:
            return np.array(image, copy=True)

    def read(self, sample_id, *, geometry=None):
        if sample_id not in self.ids:
            raise AuditError('样本不属于指定划分')
        if self.partition != 'train' and geometry is not None:
            raise AuditError('验证与预测只允许完整原图，不使用增强')
        row = self._rows[sample_id]
        image = self._read(row['image'])
        if self.partition == 'test':
            # No mask lookup, dummy target or train-directory dependency.
            pixels = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
            pixels /= np.float32(255)
            result = {'image': pixels}
        else:
            result = prepare_pair(image, self._read(row['mask'], mask=True), geometry)
        return {'id': sample_id, **result}
