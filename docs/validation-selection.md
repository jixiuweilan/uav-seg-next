# 验证、样本调度与合成选择管线

0.7.0继续实现正式运行前的管线代码，不启动训练，也不把内部指标称作官方指标。

## 已完成的代码

`uavseg.schedule` 使用显式种子产生确定性轮次顺序和共享几何变换。
每个轮次恰好覆盖全部训练ID一次，最后一个不足批量大小的批次保留。
`prepared_batches` 按计划读取审计样本，核对ID、轮次、数组类型和尺寸后堆叠；
`tensor_batches` 再复制到调用者明确指定的设备。任何阶段都不使用隐藏的全局随机状态。

`uavseg.validation.evaluate_samples` 要求验证ID有序、唯一且完整。
它逐样本检查实际ID，拒绝提前结束、乱序、缺标签和额外样本；使用既有内部口径
累计全局混淆矩阵。没有有效标签像素时不产生分数。`validate_model` 把模型目标标签
恢复到官方ID空间后调用完整图预测，并保持原始样本记录不变。

`EarliestBest` 只接受固定验证间隔、严格递增的更新点、有效内部报告和检查点摘要。
分数严格提高时替换当前候选；同分保留更早的检查点。所有输出均标记
`official_model_selection=false`，用途限定 `synthetic-selection-check`。
在划分冻结和官方指标边界确认前，不能用它开展实验模型选优。

`uavseg.runplan` 生成不可启动训练的草案，绑定审计清单、候选划分和代码摘要，
记录模型、变换、优化器、双重预算、验证间隔、指标及单检查点约束。
它只接受 `status=draft`、`frozen=false`、`training_authorized=false` 的候选划分，
生成结果也固定 `training_authorized=false`、`formal_training_command_available=false`。
必须由调用者填写批量、更新上限、批次上限和验证间隔；项目不替正式运行猜测这些数值。

## 当前验证

开发端项目隔离环境共95项合成测试通过，语法和差异检查通过。
开发端普通测试覆盖调度复现、整轮覆盖、批次读取契约、验证完整性、全局指标、
最早同分规则、固定间隔、草案身份和不可授权边界。普通测试不导入PyTorch。

执行端新增 `controls` 范围共5项：有限循环跳过后更新、全忽略批次上限、
NumPy批次到CPU张量、验证标签转换和非法标签拒绝。这些测试属于0.6/0.7新增代码，
不包含在已核验的 `b8ec92d` 基线5项报告中。

使用现有审计清单和候选划分完成一次纯JSON贯通，未读取原图：草案包含6297个训练ID、
699个验证ID，`plan_sha256=19a2f300f03c07a03648e34cb56ede53f9804faf65e3d88c0d3cbd0ef9b2075f`。
该本地草案只用1次更新/2批的合成参数核对接口，明确保留四个硬条件，不能执行正式训练。

组员后续只需在保留的 `bigdata` 环境运行一次新增范围，无需官方数据、GPU或正式训练：

```bash
git status --short
git pull --ff-only origin dev
git rev-parse HEAD
python -m uavseg.check_baseline --suite controls --output .local/baseline-controls-check-v1.json
```

如果第一条显示自己的修改，保留修改并先回报，不重置。成功标准为退出码0，
`suite=controls`、`status=passed`、`tests_run=5`，错误、失败和跳过清单均为空。
交回原始JSON和Git版本；失败时附终端错误。由负责人转达，代理负责接收、诊断和修复。

## 仍未完成的硬条件

- 当前候选划分包含尚未复核关系及证据局限，未冻结。
- 官方指标的预测0和空类别边界尚未从组织方规则确认。
- 执行端尚未测量512裁剪反向传播的峰值内存和稳定吞吐，正式批量与预算未定。
- 负责人尚未授权并启动正式训练。

因此当前没有正式训练命令。代码完成和合成测试通过都不代表训练已授权或模型有效。
