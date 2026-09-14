# Repository Agent Instructions

- Read this file, README.md, docs/requirements.md and docs/data/README.md first.
- Work only within the owner's stated phase. State the plan before action.
  Resolve routine implementation choices and investigate tool failures within
  the approved work without mechanical reconfirmation. Stop for an unresolved
  blocking prerequisite or an action outside the authorized phase.
- Teammate communication is relayed exclusively by the owner. Prepare messages
  and handoff files locally; do not contact teammates or operate external
  messaging or task services.
- This is an independent implementation. Do not import or copy historical
  project source, configs, tests, experiment narratives or model preferences.
  Initial design uses the facts here and first-party sources. Disclose any
  inherited historical knowledge instead of silently treating it as evidence.
- Commit an independent design before the owner releases historical model
  evidence. Do not search neighboring projects or account memory for model
  history during the first-layer design phase.
- This workstation is code/CPU-audit only. Do not install a training/CUDA stack
  or run GPU smoke here. Never start formal training; the owner starts any
  separately authorized formal run with a foreground command on its execution machine.
- Use official data only and one model/one checkpoint per submission. Do not
  use test labels, leaderboard feedback or test-derived pseudo-labels to fit
  or select data corrections. Check any uncertain rule before experiments.
- Treat raw data as read-only. Record confirmed corrections as versioned
  overlays after human review; never delete difficult samples automatically.
- Keep datasets, artifacts, credentials and absolute machine paths out of Git.
  Local paths belong only in ignored .local/ configuration. Use only remotes
  explicitly authorized by the owner.
- Prefer a minimal working baseline. Validate correctness before claiming
  speed or accuracy gains; compare equivalent workloads and retain provenance.
- Keep each change coherent across code, interfaces, tests and documentation.
  Run relevant available checks, report omissions, and commit only the reviewed
  phase after validation. Preserve unrelated edits. Use Chinese for HTML user
  interfaces and instructions intended for human operators and teammates, as
  requested by the owner. Preserve organizer material in its original language.
