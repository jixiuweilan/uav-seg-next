# 基线代码与执行端数值验证

日期：2026-09-15，当前版本0.5.1。已补齐内部指标、审计样本读取、小型 U-Net、
单批次更新、完整图像预测及合成检查点保存/恢复代码。
0.5.0开发端64项合成测试通过；组员回报5项数值测试中4项通过、检查点保存失败。
当前已修复文件流写入，待[定向复测及原始JSON接收](checkpoint-fix.md)。
0.5.1开发端69项合成测试通过，不包含执行端PyTorch数值测试。
不能据此宣称模型已经可训练、速度达标或取得任何准确率。

## 已实现的链路

| 模块 | 行为 | 当前验证 |
|---|---|---|
| `uavseg.metrics` | 全集混淆矩阵、Ignore与空类别边界 | 手算及合成测试通过 |
| `uavseg.samples` | 审计摘要、划分身份、文件字节及解码统计核对 | 合成 PNG 测试通过 |
| `uavseg.model` | 16/32/64/128编码器、256瓶颈、8类输出 | 语法检查通过，数值测试待执行端 |
| `uavseg.runtime` | 单批次更新、全忽略跳过、完整图预测、权重往返 | 语法检查通过，数值测试待执行端 |
| `uavseg.check_baseline` | 有限的 CPU 合成验证及 JSON 回执 | 无依赖时阻塞、禁止覆盖和输出边界测试通过 |

读取使用相对路径及私有审计快照；文件哈希、尺寸、像素哈希、标签统计须与清单一致，
拒绝符号链接。训练读取指定候选分区，可使用已记录的共享几何变换；验证仅读完整图像。
预测路径只读取测试图像，不要求训练目录或标签存在。接受候选划分用于接口验证不等于冻结或授权训练。

模型按已批准的[独立设计](independent-design.md)从零实现，不下载预训练权重。
每级两次卷积、8组 GroupNorm、ReLU；解码双线性上采样对齐跳跃连接尺寸。
损失在有效像素上计算交叉熵，全忽略批次不调用优化器更新，损失和梯度非有限时拒绝更新。
当前没有正式训练循环，也没有自动选优、断点续训或提交命令。

检查点只保存单一模型参数及来源信息，核对文件哈希、结构、数据/划分/代码摘要、
种子和指标口径后恢复；使用受限张量加载方式，不保存整个模型对象。
用途限定为 `synthetic-runtime-check`，不冒充正式训练或提交权重。
它不含优化器和随机数状态，不能恢复正式训练。

## 内部指标口径

必须显式选择 `internal-global-ignore0-v1`；这是内部接口测试口径，
**官方边界规则仍待确认，不能据此开展实验模型选优**。

- 9×9 int64矩阵，行为真实标签、列为预测；仅丢弃真实标签0。
- 真实标签非0而预测为0时，计入真实类别的漏检。
- 类别1至8分别计算IoU；并集为0时记为 `null`，不计入平均。
  真实标签没有某类、但预测出现该类时，其IoU为0并计入平均。
- 全集累计后计算均值，不先对每张图求分数再平均。
- 没有有效像素时报告无效，分数为 `null`。

例如真实标签 `[0,1,1,2,2]`，预测 `[8,1,0,1,2]`：第一项忽略，
类别1的IoU为1/3、类别2为1/2，均值5/12；其余类别记为空。

## 执行端运行方法

负责人确认没有可连接的执行端，并同意交由组员执行。
只需一名组员按[中文任务步骤](team-baseline-check.md)运行和回传，代理负责诊断与修复。
开发终端不安装 PyTorch，不运行以下数值测试。

执行端需有 Git、Python 3.11或3.12、NumPy、Pillow和可导入的 PyTorch环境。
CPU即可，无需 GPU、CUDA、官方数据、审计清单或人工判断文件。
任务说明提供已有环境检查和新建 Miniconda 环境两条路径，新环境暂定 PyTorch 2.7.1 CPU。
已有PyTorch2.8.0环境收到4/5通过的转述，完整兼容性结论待修复复测及原始报告核验。
`environment.yml` 仍是纯 CPU 审计环境，不是模型执行环境。

在执行端已有仓库中，确认没有未提交修改后更新 `dev`：

```bash
git switch dev
git pull --ff-only origin dev
git rev-parse HEAD
```

激活该机器已有的模型 Conda 环境，在仓库根目录运行：

```bash
python -m uavseg.check_baseline --output .local/baseline-runtime-check-v1.json
```

命令前台运行，固定CPU、2线程、种子0和确定性算法；不查询或分配GPU，
只生成临时合成图像和临时权重，结束后清理。运行5项数值测试：
尺寸与初始化复现、一次有效更新及全忽略跳过、非法输入和非有限梯度保护、
检查点身份与恢复、1024完整图像到PNG/ZIP。
512尺寸仅验证前向，反向使用32尺寸合成图；**不测量正式512训练的内存或速度**。

结果写入指定JSON，包含代码内容摘要、依赖版本、测试数量及失败详情。
默认运行 `suite=all`，退出码0且 `status=passed`、`tests_run=5` 才算该轮全部通过；
`--suite checkpoint`只运行1项，其通过仅代表检查点范围。1表示测试失败，2表示依赖或输入条件未满足。
缺少 PyTorch 时明确记录 `blocked`、0项数值测试，不会静默跳过。
如果依赖导入报错而未生成报告，保留终端错误文本。重复运行换 `v2` 等新文件名，不覆盖原报告。

报告保留在本地，由负责人转交或由代理从已授权执行端读取，不提交机器路径或产物到Git。
代理负责定位故障和修复，不把测试失败的诊断工作交给组员。

## 后续依赖与本机验证

数值验证通过后，再根据执行端资源完成正式运行配置与有限训练循环。
正式运行还需要冻结候选划分、确认指标边界、依据资源确定批量与更新预算。
正式训练始终由负责人另行授权并在执行端前台启动。

```bash
.conda/uav-seg-next/bin/python -m unittest discover -s tests -v
.conda/uav-seg-next/bin/python -m compileall -q uavseg execution_tests
git diff --check
```

普通测试目录有意不包含 `execution_tests`；本机通过数不包含模型数值测试。
本次没有读取官方原图、安装训练栈、运行GPU检查或正式训练。

API核对依据：[GroupNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)、
[interpolate](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html)、
[CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)、
[torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)。
这些文档不代替执行端数值验证或官方赛事评估规则。
