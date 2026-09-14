# CPU 数据审计与提交格式说明

这是独立设计评审通过后的首个实现阶段，交付 `a464e02` 已于 2026-09-09 获验收。
工具提供只读数据审计、确定性清单、参考划分检查、临时约定的平铺 PNG/ZIP 校验和打包。
不包含模型、评估器、训练循环或标注编辑器。场景筛查另见[中文复核说明](scene-review.md)。
已执行的历史检查和像素哈希解释见[验证记录](data/cpu-audit-report.md)。

以下命令供开发代理和获授权的数据机器使用。组员当前只需完成
[协作说明](team-review.md)中的浏览器体验与说明反馈，无需配置 Python。

## 项目隔离环境

在仓库根目录使用 Miniconda。环境目录 `.conda/uav-seg-next` 已被 Git 忽略，
可以直接调用其中的 Python，无需激活环境。不在此代码终端安装 CUDA 或训练框架。

```bash
conda create --prefix .conda/uav-seg-next --file environment-linux-64.lock
.conda/uav-seg-next/bin/python -m unittest discover -s tests -v
.conda/uav-seg-next/bin/python -m uavseg --help
```

[environment.yml](../environment.yml)固定直接依赖版本，其中 NumPy 为 2.5.3；
[environment-linux-64.lock](../environment-linux-64.lock)固定 Linux x86-64 构建与
SHA-256。[dependencies.json](dependencies.json)保存包地址和许可证元数据。
其他平台需要重新求解 YAML、验证后记录平台锁，不能直接使用 Linux 构建。
YAML 不固定全部传递依赖。请记录实际解释器与版本；项目外环境的失败应先检查依赖差异。

从仓库根目录执行，不需要可编辑安装。Python 标准库承担命令行、JSON、哈希和 ZIP，
Pillow 负责 PNG 解码验证，NumPy 负责标签统计。Pillow 使用 MIT-CMU，NumPy 使用
BSD-3-Clause；转发环境时须保留各包的版权与许可证，本阶段不分发这些包或模型权重。
依据：[Pillow 许可证](https://pillow.readthedocs.io/en/stable/about.html#license)、
[NumPy 许可证](https://numpy.org/doc/stable/license.html)。

## 数据访问和审计

当前本机是代码终端。下面的真实数据命令只能在有授权数据的数据机器上执行。

被忽略的 `.local/data-paths.json` 配置 `dataset_root`、
`train_images`、`train_masks`、`test_images` 和 `access: read-only`。
子目录必须是数据根内的可移植相对路径；相对数据根以配置文件所在目录解析。
机器绝对路径只放在本地配置中。程序只读这三个目录，不扫描所在项目的历史代码。
示例图片不包含在此次审计范围，也没有被允许作为额外训练数据。

```bash
.conda/uav-seg-next/bin/python -m uavseg audit \
  --config .local/data-paths.json \
  --train-split docs/data/splits/train.txt \
  --val-split docs/data/splits/val.txt \
  --all-split docs/data/splits/train_all.txt \
  --output .local/cpu-audit-v1.json
```

目录内只接受平铺普通 PNG 文件，空目录、意外文件和文件符号链接都会失败。
训练图像与掩码文件名必须完全配对。每个 PNG 都会完整验证和解码，尺寸为
1024×1024，图像是 8 位 RGB，掩码是 8 位 L，标签只能为 0–8。
调色板、动画和透明掩码会被拒绝；单 PNG 编码大小上限为 32 MiB。
工具不会自动转换、修复或跳过不合要求的样本。

三个划分文件要么全部提供，要么全部省略。每行是不带扩展名的样本编号。
重复、空白行、两侧空白、路径式编号均失败。训练与验证必须不相交，合并结果和
train_all 都必须等于全部训练样本。以解码像素哈希检查精确重复，忽略 PNG 编码差异。
重复组跨训练/验证时失败，但这不能替代同场景复核或划分冻结。

成功写入版本 1 JSON，包含：

- `manifest`：排序后的训练和测试记录、相对路径、文件字节数、模式、尺寸、
  文件 SHA-256、像素 SHA-256、九类掩码像素计数、划分编号和划分文件哈希、
  精确重复组及尚未执行的检查。
- `manifest_sha256`：规范 JSON 哈希，即 UTF-8、键排序、紧凑分隔、
  禁止 NaN、末尾一个换行。遍历顺序和机器位置不进入身份。
  划分文件原始字节变化会改变身份，即使编号集合相同。
- `producer`：工具与依赖版本，包内各 Python 源文件名和哈希组成的规范映射摘要。
  生产者信息不属于数据内容摘要。

版本 1 的 `pixel_sha256` 只计算解码后的 uint8 像素字节：从上到下逐行、
从左到右逐像素，图像按 RGB 交织，掩码每像素一个字节。
不加入模式、尺寸、文件名或 PNG 元数据，不做颜色转换和方向变换。
比较时仍应同时检查单独记录的模式与尺寸。文件 `sha256` 计算原始编码字节。
若另一清单使用同名字段，必须先核对计算定义再判断数据是否改变。

标准输出为摘要 JSON，标准错误输出进度。输入或契约错误退出码为 2，不发布清单；
遇到首个问题就停止，由负责人决定如何处理，不会静默缩小数据集。

## 验证与打包预测

以下命令要求已在其他地方生成预测，不会产生模型预测，不读取测试标签。
文件名集合以审计清单中的测试集合为准。清单摘要只能检测清单被修改，不能证明
原位置数据仍然相同；官方数据变化后须重新审计。

```bash
.conda/uav-seg-next/bin/python -m uavseg pack \
  --config .local/data-paths.json --manifest .local/cpu-audit-v1.json \
  --predictions .local/predictions --output .local/submission-v1.zip
.conda/uav-seg-next/bin/python -m uavseg validate-zip \
  --config .local/data-paths.json --manifest .local/cpu-audit-v1.json \
  --archive .local/submission-v1.zip
```

打包先验证 PNG，再按文件名排序、固定 ZIP 元数据，用 STORE 写入临时包，
发布前重新打开校验。锁定环境中相同 PNG 字节会得到相同 ZIP 字节；
重新编码即使像素相同也可能改变摘要，不宣称跨编码器字节一致。

校验不解压落盘；检查文件名集合、重复、目录、嵌套路径、符号链接、
加密、额外成员以及非法 PNG。它不评估语义精度、检查点来源或组织方是否接受。
平铺布局仍是待组织方确认的临时约定，工具没有上传命令。

## 输出保护和验证边界

所有写入拒绝覆盖已有文件、写入原始数据根、预测目录或受保护输入文件。
解析路径可防止普通符号链接别名绕过。写入失败会清理临时文件，
通过排他发布避免覆盖，输出应使用新的版本名。
这假定命令运行期间输入和目录链接保持稳定，不覆盖并发修改情形。

数据、掩码、清单、PNG、ZIP 和环境均留在被忽略的本地目录。
合成测试由开发代理执行，覆盖坏图、非法标签、路径边界、划分身份、重复图像、
ZIP 成员集合、加密标志拒绝、原子失败和确定性打包。
加密标志测试检查读取载荷前明确拒绝，不是加密算法兼容性测试。
自动验证不等于官方数据已重跑、模型已完成或训练已获授权。
