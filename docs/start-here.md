# Fresh-Session Startup Prompt

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
