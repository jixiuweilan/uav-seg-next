# 无人机航拍图像分割：独立实现

本项目依据官方规则和已核验的数据事实独立开发，不复制历史项目实现。
当前完成的是数据审计、提交格式检查、场景相似性筛查和人工复核工具。
尚未实现模型、指标评估器和训练流程。

## 组员从这里开始

请阅读[中文协作与任务说明](docs/team-review.md)。Faith 已提交旧版正式批次的部分
复核记录，已有成果继续接收，无需重新做中文版练习。无怨无悔使用
[本地原图复核页](docs/local-original-review.md)完成22条针对性核实，替代旧说明练习。
页面直接选择各人本地原图文件夹，自动核对本批文件，无需安装环境或集中上传图像。

- [场景复核操作与判断说明](docs/scene-review.md)
- [数据审计与提交格式工具说明](docs/cpu-audit.md)
- [项目要求](docs/requirements.md)与[已有数据事实](docs/data/README.md)
- [代理工作规则](AGENTS.md)与[启动说明](docs/start-here.md)

## 进度与边界

独立设计已经评审通过。首轮 CPU 工具交付 `a464e02` 于 2026-09-09 获得验收。
[历史审计记录](docs/data/cpu-audit-report.md)核对了 14,492 个范围内 PNG 的身份。
[首轮场景筛查记录](docs/data/scene-screen-report.md)包含 199 个候选和 20 个对照。
这些数字是历史批次证据，不能当作人工复核完成或无场景泄漏的证明。

两位组员的首轮代码审查已收到；后续自动验证、补测试和代码修复由开发代理承担。
人类负责实际操作体验、证据解释及判断说明的歧义反馈。
[最新接收与跟进记录](docs/data/review-followup-20260914.md)包含111条已提交判断的
结构校验结果与待核实边界。按这些判断推导的8个跨划分组尚未独立核验图像证据。
组员各自持有完整官方数据，后续复核直接使用本地副本，无需集中传图。
新划分冻结仍待核实结果就绪。正式训练由负责人在另行授权后启动。

## 开发端检查

在仓库根目录使用项目隔离环境。当前终端仅承担代码与合成测试，官方数据任务需在
获授权且持有数据的组员电脑上执行，无需中心工作站。环境建立方法见[CPU 工具说明](docs/cpu-audit.md)。

```bash
.conda/uav-seg-next/bin/python -m unittest discover -s tests -v
.conda/uav-seg-next/bin/python -m uavseg --help
git diff --check
```

代码在 `dev` 分支。数据、模型、环境、生成的练习材料与机器路径均不进入 Git。
组织方材料在 [docs/official](docs/official/README.md) 中保留原文。
