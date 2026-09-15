"""Validate an original runtime JSON against the exact tested Git revision."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from .common import AuditError, canonical, sha256, write_json


def revision_identity(commit):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise AuditError('须提供完整的40位已测试Git提交号')
    root = Path(__file__).resolve().parent.parent
    def git(*args):
        run = subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=False)
        if run.returncode:
            raise AuditError('本地不存在该提交或无法读取Git对象')
        return run.stdout
    names = git('ls-tree', '-r', '--name-only', commit, '--', 'uavseg', 'execution_tests').decode().splitlines()
    names = [name for name in names if name.endswith('.py') and name.count('/') == 1]
    if 'uavseg/check_baseline.py' not in names:
        raise AuditError('被测提交缺少基线验证入口')
    entries = {name: sha256(git('show', f'{commit}:{name}')) for name in sorted(names)}
    return sha256(canonical(entries))


def inspect_receipt(report, *, expected_source):
    if not isinstance(expected_source, str) or not re.fullmatch('[0-9a-f]{64}', expected_source):
        raise AuditError('缺少有效的预期代码摘要')
    if (not isinstance(report, dict) or report.get('kind') != 'synthetic-baseline-runtime-check' or
            type(report.get('schema_version')) is not int or report['schema_version'] != 1):
        raise AuditError('运行报告类型或版本无效')
    if report.get('source_sha256') != expected_source:
        raise AuditError('报告代码摘要与被测Git提交不一致')
    if (report.get('device') != 'cpu' or type(report.get('seed')) is not int or report['seed'] != 0 or
            report.get('official_data_read') is not False or report.get('formal_training_started') is not False):
        raise AuditError('报告执行范围与合成CPU任务不一致')
    scope = report.get('suite', 'all')  # 0.5.0 reports predate explicit suite labels.
    expected_tests = {'all': 5, 'checkpoint': 1}.get(scope)
    if expected_tests is None:
        raise AuditError('未知报告测试范围')
    count = report.get('tests_run')
    if type(count) is not int or not 0 <= count <= expected_tests:
        raise AuditError('报告测试数无效')
    status = report.get('status')
    if status == 'blocked':
        if count != 0 or not isinstance(report.get('reason'), str) or not report['reason']:
            raise AuditError('阻塞报告必须说明原因且未执行测试')
    elif status in ('passed', 'failed'):
        for key in ('python_version', 'torch_version', 'numpy_version', 'pillow_version'):
            if not isinstance(report.get(key), str) or not report[key].strip():
                raise AuditError('运行报告缺少依赖版本')
        problems = []
        for key in ('failures', 'errors', 'skipped'):
            value = report.get(key)
            if not isinstance(value, list):
                raise AuditError('运行报告缺少错误和跳过清单')
            problems.extend(value)
        if status == 'passed' and (count != expected_tests or problems):
            raise AuditError('通过状态与测试数或错误清单矛盾')
        if status == 'failed' and (count == 0 or not problems or len(problems) > count):
            raise AuditError('失败状态与执行明细矛盾')
    else:
        raise AuditError('未知运行状态')
    return {'reported_status': status, 'suite': scope, 'tests_run': count,
            'source_matches_tested_commit': True,
            'full_suite_reported_passed': status == 'passed' and scope == 'all',
            'execution_independently_observed': False, 'reporter_identity_authenticated': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='核对原始合成验证JSON与被测Git版本；不运行模型或读取官方数据')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(__file__).resolve().parent.parent
        if not args.output.resolve().is_relative_to((root / '.local').resolve()):
            raise AuditError('接收回执须位于仓库 .local 下')
        # Validate and hash the same bytes; never regenerate a missing raw report.
        raw = args.report.read_bytes()
        report = json.loads(raw)
        result = inspect_receipt(report, expected_source=revision_identity(args.commit))
        receipt = {'kind': 'runtime-report-receipt', 'tested_commit': args.commit,
                   'report_file_sha256': sha256(raw), 'report_file_bytes': len(raw), **result}
        write_json(args.output, receipt, [args.report])
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    except (AuditError, OSError, ValueError) as exc:
        print(f'运行报告接收失败：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
