# UAV Segmentation — Independent Start

Build an original, reproducible and rules-compliant semantic-segmentation
solution for low-altitude aerial imagery. This repository starts with verified
requirements and data facts, not a preselected model or implementation stack.

## Read first

1. [Agent instructions](AGENTS.md)
2. [Requirements and open decisions](docs/requirements.md)
3. [Data facts and access](docs/data/README.md)
4. [Fresh-session startup prompt](docs/start-here.md)

The organizer source is preserved byte-for-byte under [docs/official/](docs/official/README.md).
Its method suggestions are suggestions, not project results or required choices.

## Current scope

The [independent minimal design](docs/independent-design.md) was committed before
the owner approved the first [CPU audit/format increment](docs/cpu-audit.md).
This increment provides data/pair auditing, deterministic content manifests,
reference split checks, and grayscale PNG/ZIP validation and packaging.
The [validation record](docs/data/cpu-audit-report.md) includes 19 passing
synthetic tests and the verified identities of all 14,492 scoped PNGs.
No model, metric evaluator or training loop exists yet. The local branch is
`dev`; there is no remote or upstream. No earlier implementation was copied.
Raw data and artifacts stay outside Git.

The owner accepted this increment at `a464e02` on 2026-09-09 and released the next
[training scene screening and human-review increment](docs/scene-review.md).
It adds reproducible similarity candidates, an offline review page and checks
for confirmed groups crossing the reference split. The [initial evidence](docs/data/scene-screen-report.md)
includes 31 passing tests and a first review batch of 199 candidates plus 20
controls from the full training collection. Actual review decisions and
a new split freeze remain pending; model implementation and execution are later
phases.

## Local setup and checks

The owner selected an isolated Miniconda environment at `.conda/uav-seg-next`.
See the [CPU guide](docs/cpu-audit.md) for setup, command contracts and limits.
Machine-specific data paths are in ignored `.local/data-paths.json`.

```bash
git status --short
git branch --show-current
git remote -v
git diff --check
.conda/uav-seg-next/bin/python -m unittest discover -s tests -v
.conda/uav-seg-next/bin/python -m uavseg --help
```

The environment contains CPU audit dependencies only. These checks do not
validate a model or training runtime.
