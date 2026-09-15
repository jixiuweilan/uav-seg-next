# Fresh-Session Startup Prompt

## Current continuation

最新负责人指示：结束两处图像依据追问，继续项目，不再给组员追加这轮工作。
当前已交付[基线数据准备与候选划分](baseline-data.md)，包含纯 CPU 的同步几何变换、
标签映射和只读 JSON 的整组候选划分。沟通只经负责人转达；查看当前 README 和 Git 状态。
当前已继续交付[基线代码与执行端验证入口](baseline-runtime.md)：64项本机合成测试通过，
0.5.1修复版本全量5/5通过的v2原始JSON已接收，代码摘要与被测提交完全一致，不再安排旧检查点复测。
0.6.0继续交付[有限更新控制与报告接收工具](training-control.md)，新增模型适配器数值测试尚未运行。
0.7.0继续交付[验证、确定性调度和合成选择管线](validation-selection.md)，
执行端待一次性运行新增的5项 `controls` 测试。
候选未冻结，正式训练入口尚未交付；不在当前终端安装训练栈或读取官方原图。
已授权工作能够推进时必须继续，不因子任务、测试或提交完成就等待负责人再次说“继续”。
负责人确认无可连接执行端，同意由一名组员完成数值验证；
代码与[中文操作步骤](team-baseline-check.md)已经备齐，沟通仍由负责人转达。

The independent design was committed as `9d7dc61` and subsequently approved.
The owner released the first CPU audit/format increment and selected Miniconda
for the project-local `.conda/uav-seg-next` environment. Start with the current
README and [CPU guide](cpu-audit.md); inspect Git state and recorded validation
evidence before making any status claim. The owner accepted the first increment
at `a464e02` on 2026-09-09 and then released the
[scene screening/review increment](scene-review.md). Resolve routine details
within that work without asking the owner to redefine the scope. Do not repeat
completed reviews or first-layer design. Human scene decisions, split freeze,
label corrections and execution authorization must retain their evidence boundaries.

> Read AGENTS.md, README.md, docs/requirements.md, docs/data/README.md,
> docs/independent-design.md, docs/cpu-audit.md and docs/scene-review.md. Use the
> accepted CPU audit and the released scene-review tooling within the owner's
> current request; inspect current artifacts before claiming completed reviews.
> Use the project-local Miniconda Python for checks. Preserve raw data and local
> state; do not inspect historical model evidence, install a training stack,
> run GPU smoke, start training or create a remote. Continue authorized code work
> and available validation. Stop only for completion, an owner stop request or a
> concrete prerequisite that cannot be resolved within the approved boundaries.

## Original independent-design prompt (historical)

The following prompt records how the independent design phase was initiated.
It does not restart that completed phase during a current continuation.

Open this repository as the workspace in a new conversation. Do not attach
historical project conversations, model recommendations or experiment ledgers.
Copy the following prompt:

> Read AGENTS.md, README.md, docs/requirements.md, docs/data/README.md and the
> source documents they reference. Design an original minimal UAV semantic
> segmentation implementation from these requirements and verified data facts.
> Do not inspect historical repositories or import their implementation,
> model rankings, configs, tests or experimental conclusions. If you already
> have historical model knowledge from memory, disclose that limitation and
> do not use it as a design premise. First record the requirement/evidence
> boundary, a small end-to-end architecture, data-quality workflow, dependencies
> requiring approval, CPU-testable acceptance gates, execution-machine resource
> checks for later authorization, and the first implementation increment.
> Include genuinely open decisions, not a broad experiment matrix. Commit this
> independent design after available checks, then return for review before
> implementation or historical-evidence disclosure. This machine must not train
> or run GPU smoke; no training stack, cloud service or remote repository is
> authorized. Do not assume fresh code is more efficient without measurement.

A fresh conversation and independent Git repository are procedural separation,
not a guarantee that account memory or filesystem access has been erased.
