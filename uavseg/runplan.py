"""Create a non-authorizing, identity-bound baseline plumbing plan."""

import argparse
import json
from pathlib import Path
import re
import sys

from .common import AuditError, canonical, read_json, sha256, write_json
from .metrics import INTERNAL_POLICY
from .contracts import MODEL_ID
from .samples import candidate_ids, validate_document
from .training import Budget


def draft_plan(document, proposal, *, source_sha256, batch_size, budget, validation_cadence):
    validate_document(document)
    train = candidate_ids(document, proposal, 'train')
    val = candidate_ids(document, proposal, 'val')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise AuditError('运行草案须绑定有效代码摘要')
    if type(batch_size) is not int or not 1 <= batch_size <= len(train):
        raise AuditError('批量大小须为1至训练样本数的整数')
    if not isinstance(budget, Budget):
        raise AuditError('运行草案须提供双重有限预算')
    if (type(validation_cadence) is not int or validation_cadence <= 0 or
            validation_cadence > budget.max_updates):
        raise AuditError('验证间隔须为不超过更新上限的正整数')
    split = proposal['split']
    if (split.get('status') != 'draft' or split.get('frozen') is not False or
            split.get('training_authorized') is not False):
        raise AuditError('当前生成器只接受未冻结、未授权的候选划分用于合成管线草案')
    plan = {'schema_version': 1, 'kind': 'baseline-run-plan', 'status': 'draft',
            'purpose': 'synthetic-plumbing-only', 'training_authorized': False,
            'formal_training_command_available': False,
            'model': {'id': MODEL_ID, 'initialization': 'random', 'classes': 8},
            'data': {'manifest_sha256': document['manifest_sha256'],
                     'split_sha256': proposal['split_sha256'],
                     'split_status': split['status'], 'split_frozen': split['frozen'],
                     'train_samples': len(train), 'validation_samples': len(val),
                     'crop_size': [512, 512], 'validation_size': [1024, 1024]},
            'optimization': {'optimizer': 'AdamW', 'learning_rate': 0.001,
                             'weight_decay': 0.0001, 'seed': 0,
                             'batch_size': batch_size,
                             'max_updates': budget.max_updates,
                             'max_batches': budget.max_batches,
                             'validation_cadence_updates': validation_cadence},
            'metric': {'policy': INTERNAL_POLICY, 'official_equivalence_confirmed': False,
                       'experimental_selection_allowed': False},
            'checkpoint': {'one_model': True, 'one_selected_checkpoint': True,
                           'tie_break': 'earliest', 'averaging': False},
            'source_sha256': source_sha256,
            'unresolved_gates': ['frozen_split', 'official_metric_edges',
                                 'execution_resource_budget', 'owner_training_authorization']}
    return {'plan': plan, 'plan_sha256': sha256(canonical(plan))}


def main(argv=None):
    parser = argparse.ArgumentParser(description='生成不可启动正式训练的基线管线草案，只读取审计和候选划分JSON')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('--max-batches', type=int, required=True)
    parser.add_argument('--validation-cadence', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(__file__).resolve().parent.parent
        if not args.output.resolve().is_relative_to((root / '.local').resolve()):
            raise AuditError('运行草案须写入仓库 .local 目录')
        result = draft_plan(read_json(args.manifest), read_json(args.split),
                            source_sha256=args.source_sha256, batch_size=args.batch_size,
                            budget=Budget(args.max_updates, args.max_batches),
                            validation_cadence=args.validation_cadence)
        write_json(args.output, result, [args.manifest, args.split])
        print(json.dumps({'status': 'draft', 'plan_sha256': result['plan_sha256'],
                          'training_authorized': False}, ensure_ascii=False))
        return 0
    except (AuditError, OSError, ValueError) as exc:
        print(f'运行草案生成失败：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
