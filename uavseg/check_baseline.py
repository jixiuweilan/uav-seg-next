"""Foreground, CPU-only synthetic numerical checks for an execution machine."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import unittest

from .common import AuditError, canonical, check_output, sha256, write_json


def source_identity():
    root = Path(__file__).resolve().parent.parent
    files = sorted([*root.glob('uavseg/*.py'), *root.glob('execution_tests/*.py')])
    entries = {path.relative_to(root).as_posix(): sha256(path.read_bytes()) for path in files}
    return sha256(canonical(entries))


def main(argv=None):
    parser = argparse.ArgumentParser(description='在执行端前台运行 CPU 合成数值验证；不读取官方数据、不运行正式训练')
    parser.add_argument('--output', type=Path, required=True, help='仓库 .local 下的新版本 JSON 报告')
    args = parser.parse_args(argv)
    try:
        local = Path(__file__).resolve().parent.parent / '.local'
        if not args.output.resolve().is_relative_to(local.resolve()):
            raise AuditError('报告须写入仓库 .local 目录')
        check_output(args.output, [])
        report = {'schema_version': 1, 'kind': 'synthetic-baseline-runtime-check',
                  'source_sha256': source_identity(), 'python_version': platform.python_version(),
                  'device': 'cpu', 'seed': 0, 'official_data_read': False,
                  'formal_training_started': False, 'tests_run': 0}
        if importlib.util.find_spec('torch') is None:
            report.update(status='blocked', reason='缺少 PyTorch；请在已有模型环境的执行端运行，本机不要安装训练栈')
            write_json(args.output, report, [])
            print(json.dumps(report, ensure_ascii=False))
            return 2
        import torch
        import numpy as np
        from PIL import __version__ as pillow_version
        from execution_tests import test_baseline

        # No CUDA query or allocation. These checks explicitly use CPU tensors.
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_baseline)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
        report.update(status='passed' if passed else 'failed', tests_run=result.testsRun,
                      torch_version=str(torch.__version__), numpy_version=np.__version__,
                      pillow_version=pillow_version, cpu_threads=2,
                      failures=[{'test': test.id(), 'detail': detail} for test, detail in result.failures],
                      errors=[{'test': test.id(), 'detail': detail} for test, detail in result.errors],
                      skipped=[test.id() for test, _ in result.skipped])
        write_json(args.output, report, [])
        print(json.dumps(report, ensure_ascii=False))
        return 0 if passed else 1
    except (AuditError, OSError, ImportError, RuntimeError) as exc:
        print(f'合成验证未完成：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
