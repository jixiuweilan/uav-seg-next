# 检查点保存失败：接收记录与定向复测

**最新：负责人已转达修复版本 `b8ec92df7c84809098f3ecd956c86e0989bd69ca`
全量5/5通过、退出码0。组员本次跑了全部5项，报告为 `.local/baseline-runtime-check-v2.json`，
并保留v1。复测任务结束，不需要再执行下方旧命令。**
环境仍为Python3.11.15 / PyTorch2.8.0+cu128 / NumPy2.0.2 / Pillow11.3.0，WSL2 Ubuntu。
当前仅接收文字回报，两个原始JSON仍待转交。[后续代码与接收方法](training-control.md)。

以下保留首轮失败及当时复测安排，作为修复历史。

负责人已转达组员在 `a556b630889aed3d07d104089e787dc3b5a28802` 上的运行摘要：
5项中4项通过，检查点往返测试报 `RuntimeError: invalid file name`。
已完成的4项保留，不要求整轮重做。当前仅收到文字，原始JSON及其内容摘要尚未核验；
组员报告该文件已生成、大小1711字节，不能将此转述视为原始报告已接收。

| 项目 | 组员报告值 |
|---|---|
| 系统 | Linux 6.6.87.2-microsoft-standard-WSL2，x86_64 |
| 环境 | bigdata |
| Python | 3.11.15 |
| PyTorch | 2.8.0+cu128 |
| NumPy / Pillow | 2.0.2 / 11.3.0 |
| 通过 | 完整图像PNG/ZIP、非法输入和梯度保护、尺寸和种子复现、有效更新及全忽略跳过 |
| 未通过 | 检查点保存、恢复及身份拒绝测试 |

## 修复依据与边界

原实现把 `.uavseg-随机字符` 临时路径直接传给 `torch.save`。
PyTorch 2.8的文件名写入器去掉最后一个点及后面的字符后，得到空归档名并拒绝保存。
这与组员提供的错误吻合，不是要求组员更换Python才能解决的问题；也不能概括成所有隐藏文件均不可保存。
依据为[PyTorch 2.8归档写入源码](https://github.com/pytorch/pytorch/blob/v2.8.0/caffe2/serialize/inline_container.cc)
及[文件流序列化源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/serialization.py)。

0.5.1改为打开临时二进制文件流，再交给 `torch.save`，使用文件流写入器。
全部写完并关闭文件后才发布目标；异常清理临时文件，已有文件和受保护目录仍拒绝写入。
本机新增无PyTorch的写入保护及定向测试选择检查，69项合成测试全部通过，语法及差异检查通过；
这些检查不包含PyTorch数值执行，实际模型往返仍需执行端复测。

Python3.12是原任务的环境建议，当前代码未硬性检查该版本。接收这次3.11环境结果，
并在同一环境复测，以减少同时变化的条件；不宣称已验证所有3.11或3.12环境。
已有CUDA构建的PyTorch无需卸载，验证命令仍只运行CPU，不安装或调用GPU环境。

## 请负责人转给原执行组员

保留原仓库、`bigdata`环境和v1报告，不重新克隆、不重建环境。在现有仓库根目录逐条运行；
若状态检查显示自己的修改或更新失败，先停在该条并回传提示：

```bash
conda activate bigdata
git status --short
git pull --ff-only origin dev
git rev-parse HEAD
python -m uavseg.check_baseline --suite checkpoint --output .local/baseline-checkpoint-recheck-v1.json
```

如果 `git status --short` 显示自己修改的文件，或更新报冲突，保留修改并回传提示，不重置仓库。
命令只跑1项检查点测试，包含保存、恢复、来源核对、禁止覆盖及中断写入清理。
预期报告为 `suite=checkpoint`、`status=passed`、`tests_run=1`。
它仅表示本次检查点复测通过，不会被记成新版本全部5项通过。

请交回以下内容，由负责人转给代理：

1. 原 `.local/baseline-runtime-check-v1.json` 文件。
2. 新 `.local/baseline-checkpoint-recheck-v1.json` 文件。
3. 新Git版本号；失败时附终端错误。没生成报告时直接注明。

文件已存在时改用新编号，不删除旧报告。代理负责核对两轮证据、诊断和修复；组员无需自行改代码。
