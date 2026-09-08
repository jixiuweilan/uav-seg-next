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

## Scope of this starting point

Only documents, source provenance and split references are versioned. The
[independent minimal design](docs/independent-design.md) is recorded for owner
review. No model, training loop, inference engine, dependency stack or runtime
tests exist yet.
The local branch is `dev`; there is no remote or upstream. No code from an
earlier implementation has been copied. Raw data stays outside the repository.

Next: review the independent design and release the first CPU audit/format
increment if accepted. Implementation and training are later phases; stronger
Agent capability is not itself evidence that a rewrite will be faster or more
accurate.

## Local setup and checks

No installation is required to read this starting point. Use standard Python 3
to inspect JSON and Git for source checks; do not install training dependencies
on this code-only workstation. Machine-specific paths, if configured, are in
ignored `.local/data-paths.json`. They are not a portable setup contract.

```bash
git status --short
git branch --show-current
git remote -v
git diff --check
python3 -m json.tool docs/data/summary.json
```

When implementation starts, add only the dependencies and tests required by
the chosen first capability. Do not carry over a historical environment or
pretend that these document checks validate a nonexistent training runtime.
