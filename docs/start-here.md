# Fresh-Session Startup Prompt

## Current continuation

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
> run GPU smoke, start training or create a remote. Report unperformed gates
> explicitly and wait at the current phase's review boundary.

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
