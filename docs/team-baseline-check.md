# 组员任务：运行一次基线合成验证

**本轮组员已回报修复版本全量5/5通过，运行任务结束。**
请只转交已生成的v1、v2原始JSON，保留现有环境和文件，不重新安装或运行测试。
以下为首次任务的历史操作参考，最新状态见[接收记录](checkpoint-fix.md)。

负责人已同意由组员执行。**只需要一名组员接这项任务**：优先选择已装 Miniconda、
能使用终端和 Git 的人；其他组员这轮无需操作。成功或失败都如实交回，代码诊断和修复由开发代理完成。

目的：验证刚实现的模型能正确前向、更新参数、保存恢复权重并生成合法PNG/ZIP。
本次仅使用自动生成的假图，CPU即可，不读取你手中的官方数据，不需要显卡或安装CUDA，
不做正式训练。不需要SSH，也不需要把电脑或数据交给代理。

## 1. 获取本轮代码

Windows打开 **Miniconda Prompt**，Linux/macOS打开能运行 `conda` 的终端。
进入你平时存放项目的文件夹，执行以下命令，使用新文件夹避免影响旧工作：

```bash
git clone --branch dev --single-branch https://github.com/jixiuweilan/uav-seg-next.git uav-seg-baseline-check
cd uav-seg-baseline-check
git rev-parse HEAD
```

记下最后一条输出的版本号。若已有同名文件夹，不删除它，给新文件夹换个名称。
本轮代码应包含 `uavseg/check_baseline.py` 和本说明。没有 `git` 或 `conda` 命令时，
将报错交给负责人，不自行绕过或重复旧复核任务。

## 2. 准备隔离环境

**已有可用的模型 Conda 环境**：激活它，用下面命令检查依赖。

```bash
python -c "import sys, torch, numpy, PIL; print(sys.version); print(torch.__version__); print(numpy.__version__); print(PIL.__version__)"
```

新环境建议Python3.12；已有3.11环境可先运行并如实记录版本。PyTorch、NumPy、Pillow均须能导入。
满足时直接进入第3步，
不修改原环境；不满足时使用下面的新环境方案。

**没有可用环境**：在本轮新仓库根目录创建项目独立环境。

```bash
conda create --prefix ./.conda/uav-seg-runtime -c conda-forge python=3.12 numpy=2.2 pillow=11.2 pip -y
conda activate ./.conda/uav-seg-runtime
```

Windows或Linux接着执行CPU版安装：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

Apple芯片的macOS执行以下命令代替上面那条：

```bash
python -m pip install torch==2.7.1
```

随后再运行本节开头的依赖检查命令。每条命令成功后才执行下一条。
网络下载失败、提示没有对应安装包、环境创建失败时，直接交回出错命令和终端文字；
不换模型、不安装GPU驱动、不修改已有环境。其他系统/架构先回报系统类型。

安装方法依据[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/)
及[官方CPU安装包目录](https://download.pytorch.org/whl/cpu/torch/)。2.7.1是本轮固定的
合成验证候选版本，不代表最新版本或已经通过本项目验证；实际依赖版本会写进报告。

## 3. 运行一次验证

确认终端仍在仓库根目录、已激活上述环境，执行：

```bash
python -m uavseg.check_baseline --output .local/baseline-runtime-check-v1.json
```

等命令结束，过程中会逐项显示5个测试结果。无需自己分析测试名或修改代码。
成功时最后的JSON应包含 `"status": "passed"` 和 `"tests_run": 5`。
失败也是有效反馈，不需要反复尝试直到通过。

若程序意外退出、长时间没有新输出或被系统关闭，记录最后停在哪一项并交回终端文字。
报告若已存在，下次使用 `baseline-runtime-check-v2.json` 等新名称，不删除旧报告。

## 4. 交回给负责人

请发这三项，由负责人转给开发代理：

1. `.local/baseline-runtime-check-v1.json` 文件；若没生成，注明“没有报告”。
2. 第1步输出的Git版本号，以及操作系统名称。
3. 失败时附出错命令及完整终端错误文字。无需发官方图片、数据文件或整个环境。

完成以上交回后，本任务结束，不接着运行正式训练。代理负责核验代码摘要、
分析数值结果、修复代码并决定后续步骤；不要求组员做代码审查或故障诊断。
