# Independent Minimal Design

Date: 2026-09-08. Starting revision: `e63aa7e`; independent design committed as
`9d7dc61` before implementation. The owner subsequently approved the design and
first CPU increment and selected Miniconda for its isolated environment. See
the [implementation guide](cpu-audit.md) for current scope. Later model and
execution phases are not released. The design decisions below are retained.

## 1. Evidence and independence

The design uses [requirements](requirements.md), the unchanged
[organizer snapshot](official/README.md), [data facts](data/README.md),
[summary](data/summary.json), and the first-party references below. The session
arrived with an account-level summary mentioning historical model choices and
experiments. That exposure cannot be undone; none is used to choose this design.
No memory files, neighboring source, historical configurations, tests, model
rankings or experiment records were inspected or imported. Independence here
means a separately reasoned implementation, not a claim of a novel architecture.

The snapshot defines nine label IDs, with 0 ignored and eight evaluated classes,
official-data-only use, and grayscale PNG predictions in a ZIP. Owner policy
further fixes one model/one checkpoint and forbids checkpoint/seed averaging.
The source's architecture suggestions and test-distribution table are not
performance evidence or inputs to class weighting, cleaning or model selection.
The snapshot mentions reproduction after both the second and semifinal stages;
the current stage notice must settle the actual deadline.

Recorded audit evidence: 6,996 labeled pairs, 500 Test 1 images, 11 example pairs;
all recorded PNGs decode at 1024 x 1024 with valid modes/IDs. The 6,297/699 split
is an identity reference, not a verified independent holdout. Near duplicates,
scene leakage, annotation correctness and train-test overlap remain unchecked.
The inventory and split hashes were verified for this design, but images were
not decoded again. Example pairs are excluded from model training and selection.
Test 2/3 and source-flight metadata are unavailable in the recorded audit.

No package, pretrained weight, annotation service or container is installed in
this phase. No training, GPU smoke, remote creation or upload is authorized.
All runtime artifacts and machine paths stay in ignored local storage. Versioned
manifests contain dataset-relative references and hashes, never raw imagery.

## 2. One end-to-end baseline

```mermaid
flowchart LR
    A[Read-only official data] --> B[Audit and content manifest]
    B --> C[Reviewed groups and frozen split]
    C --> D[Train on execution machine]
    D --> E[One selected checkpoint]
    E --> F[Full-image validation or inference]
    F --> G[Grayscale PNG and ZIP validator]
    C --> H[Human annotation review]
    H --> I[Versioned approved overlays]
    I --> D
```

Use a small Python package with explicit functions, a standard-library CLI and
JSON configuration. Proposed module boundaries are `data` (inventory, split,
overlay resolution, paired transforms), `metrics`, `model`, `train`, `predict`
and `submission`. Share label mapping and preprocessing between entrypoints.
An inference entrypoint reads images and one checkpoint; it never loads masks.
There is no plugin registry, distributed runner or experiment service initially.

**Model decision.** Implement a compact U-Net-style convolutional encoder and
decoder afresh. The contracting/expanding paths and skip connections are an
established localization design, not a demonstrated aerial-data advantage.
[U-Net paper](https://arxiv.org/abs/1505.04597v1).

- Four encoder stages at widths 16, 32, 64, 128; 2 x 2 max pooling after each;
  a 256-channel bottleneck; four decoder stages at widths 128, 64, 32, 16.
- Each stage has two padded 3 x 3 convolutions, each followed by GroupNorm with
  eight groups and ReLU. Decoder stages bilinearly resize to the corresponding
  skip size with `align_corners=False`, concatenate the skip, then convolve.
  GroupNorm uses input statistics in both training and evaluation; choosing it
  avoids a running-statistics dependency for the proposed small batches.
  [GroupNorm API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html).
- A 1 x 1 head emits eight logits per pixel. Raw labels 1--8 map to targets
  0--7; raw Ignore 0 maps to sentinel -100 before loss. Argmax plus one restores
  official IDs. Ignore is an evaluation mask, not a learned output class.
- Random initialization, no pretrained downloads. This avoids a weight-source
  dependency in the first baseline but may limit accuracy and require more
  optimization. The model is selected for implementation simplicity; no size,
  throughput, convergence or competitive-score claim has been measured.

**Data and optimization proposal.** RGB bytes become float32 CHW values divided
by 255; no test-derived normalization. Training samples use uniform 512 x 512
crops at native pixel scale, paired horizontal/vertical flips and quarter-turns.
Record each geometric draw once and apply it to both image and mask. Do not
discard crops for being difficult; all-ignore crops have a separate no-update
handling rule. Validation/inference use complete 1024 x 1024 images, without
augmentation or resizing. Crops reduce training context, a declared limitation.

Use unweighted cross entropy over valid pixels and AdamW, with initial proposed
learning rate 0.001 and weight decay 0.0001, one seed (0), and a constant learning
rate. These are design defaults, not tuned results. Batch size and finite update
budget require execution-machine evidence and owner approval before a run.
No class balancing or additional loss is justified by a baseline result yet.
The loss receives logits and integer targets; explicitly handle an all-ignore
batch by skipping the optimizer update and logging it. The run budget must also
bound consumed batches so repeated skipped updates cannot extend a run forever.
[CrossEntropyLoss API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

Select the earliest checkpoint with the highest frozen-validation mIoU at a
predeclared validation cadence within the approved budget. Keep one selected
checkpoint for prediction, with no averaging or test-score-based replacement.
Final training on all labeled data would be a separately reviewed change.

**Metric contract proposed for review.** Accumulate one int64 9 x 9 confusion
matrix over the whole holdout in official ID space, rows = GT and columns =
prediction. Discard only GT=0 pixels. A prediction of 0 on valid GT remains in
column 0 and contributes a false negative to that GT class. For each class 1--8,
IoU = diagonal / (row sum + column sum - diagonal). A zero-union class is reported
as null and excluded from the macro average; a class absent in GT but predicted
has nonzero union and scores zero. Report all class supports and the number of
classes averaged. No valid pixels means an invalid evaluation, not a perfect
score. These edge rules and global aggregation require organizer confirmation
before calling this an official evaluator or selecting an experimental model.

**Submission contract.** Produce uint8 mode-L PNGs with exact original basenames,
1024 x 1024 dimensions and IDs 0--8 (this baseline predicts 1--8). Reopen every
PNG and validate the entire archive against the selected test-image manifest:
one entry per expected name, no duplicate, missing or extra members. A flat ZIP
is the proposed layout, pending organizer confirmation. Reject palette, RGB,
wrong size and invalid IDs rather than silently converting them. Runtime
provenance stays beside the ZIP, not as extra archive members. Pillow distinguishes
8-bit grayscale L from palette P explicitly.
[Pillow modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes).

## 3. Data quality without changing the evaluation target

1. **Reproducible audit.** Decode image/mask pairs, require RGB/L and 1024 x 1024,
   check IDs and exact filename pairing, and compute raw-byte and decoded-pixel
   SHA-256 identities. Emit deterministic records sorted by sample ID, including
   class counts and schema version. Fail on corruption or contract violations;
   report affected IDs and wait for direction, preserving files.
2. **Group before splitting.** Recheck exact decoded duplicates. Screen training
   images for near duplicates with fixed grayscale thumbnail perceptual hashes
   and normalized thumbnail distances, including quarter-turn/flip comparisons.
   Use these only to shortlist pairs; review full-resolution pairs for shared
   scene, adjacent frames and overlapping crops. Request flight/source metadata
   if available. Record method/version, scores, reviewed edges, reviewer and
   confirmed/rejected/uncertain status. Components of confirmed same-scene edges
   become indivisible groups; review large components for erroneous chaining.
   Thumbnail matches cannot establish all scene relationships.
3. **Freeze a holdout.** Target approximately 10% of the official labeled pool,
   assigning whole reviewed groups with a fixed seed and inspecting training-label
   class coverage. Freeze group records, split IDs, allocation policy and hashes
   before model fitting. Never split a confirmed group just to reach 699 images.
   If grouping/coverage is inadequate, report the limitation and ask the owner
   to resolve it before releasing training; the reference split may support
   plumbing checks but cannot be presented as leakage-cleared. Do not repeatedly
   search seeds for a favorable score. A later train-test image-only overlap audit
   may flag provenance questions; it cannot tune cleaning or model selection.
4. **Rank suspicion after a baseline exists.** Initially provide image/mask
   overlays and boundary views. Later use group-held-out or group-aware out-of-fold
   predictions over the training partition for annotation triage. For each such
   prediction, retain its checkpoint and training-ID exclusion proof. Rank
   disagreement fraction, directional class confusion, confident missing/extra
   regions, and boundary displacement/alignment separately. Visual edge mismatch
   and weak-model disagreement are cues, not proof. Out-of-fold fitting needs its
   own execution approval; triage checkpoints are never combined for submission.
5. **Human review and overlays.** Review high-ranked samples per signal/class and
   a seeded random sample from the low-ranked remainder. Predeclare review counts
   and sampling strata; record confirmed, rejected and uncertain suspicions,
   reasons, reviewer identity and timestamp. Report confirmation rates separately
   for ranked and control samples, without claiming zero missed errors. A confirmed
   correction references original image/mask hashes, corrected mask hash, reason
   and annotation-tool/version if used. Human approval precedes activation.
   Version overlay metadata in Git; original and corrected masks remain outside
   Git. Reject stale original hashes, duplicate active corrections or unresolved
   decisions. Disabling the overlay must restore the original effective dataset.
6. **Keep comparisons interpretable.** Freeze evaluation labels before fitting.
   Do not use candidate predictions to improve holdout labels. If an independently
   confirmed evaluation error requires repair, create a new evaluation version
   and rescore all compared checkpoints on it, retaining old-version results.
   Compare training-overlay changes with the model, split, seed, update budget
   and evaluation labels fixed. Compare model changes with the dataset version
   fixed. No test labels, leaderboard feedback or test pseudo-labels enter review;
   no automatic correction, hard-sample removal or confidence-based deletion.

## 4. Minimal dependencies and provenance

| Capability | Proposed dependencies | Release condition |
| --- | --- | --- |
| Current design checks | Existing Python 3 standard library and Git | Authorized and used; no installation |
| First CPU audit/format increment | Pillow and NumPy; standard-library unittest | Owner approves increment and dependency versions/licenses before installation |
| Model implementation and numerical gates | PyTorch plus the audit dependencies | Later explicit scope approval; CPU numerical gates only in an approved environment, no training-stack installation here |
| Training and GPU resource validation | Compatible PyTorch build and execution-machine driver/runtime | Separate execution-machine authorization; no local CUDA setup |
| Confirmed mask repair, Docker and technical PDF | Tool selection deferred until needed | Separate review of tool, data handling, versions and stage requirements |

Pin the actually reviewed versions and package hashes in the first relevant
increment, with first-party source/license URLs and dependency notices. Browsed
API documentation is not a package compatibility test or an installation pin.
No pretrained weight is selected. Any future proposal needs a public academic
source, separate code/weight license checks, pretraining-data provenance and a
reviewed answer to test overlap; unresolved eligibility blocks its use.

Use canonical JSON (UTF-8, sorted keys, defined separators, no machine paths)
and stable manifest ordering for identities. A dataset version binds base
manifest, split/group records, overlay metadata and effective mask hashes.
A run binds that version, resolved portable config, code commit, environment
lock, seed, optimizer state, RNG states, update count and metric policy. Seed
the sampler, workers and transforms explicitly and retain their resume state. Keep
epoch-boundary resume sufficient initially; do not promise exact mid-epoch
replay. Record checkpoint file SHA-256 separately from a canonical tensor-content
identity (sorted keys, dtype, shape and bytes), since serialization bytes need
not be stable. Paths resolve only through ignored `.local/` configuration.
Deterministic replay is checked within a pinned environment, not promised across
devices/releases. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## 5. Acceptance gates defined at the design commit

| Gate | Hand-checkable evidence and pass condition | Where/when |
| --- | --- | --- |
| Audit and identity | Synthetic corrupt PNG, missing pair, wrong mode/size, ID 9 and duplicate ID must fail with IDs; stable manifests survive file enumeration order changes; one changed mask changes dataset identity | First CPU increment |
| Leakage and split | Identical pixels in differently encoded PNGs match; transformed near-duplicate fixture is shortlisted; a known same-scene group crossing the split fails; all IDs covered once, no train/val overlap; uncertain real grouping is not marked cleared | CPU; real split freeze needs human review |
| Metrics | GT `[0,1,1,2,2]`, prediction `[8,1,0,1,2]`: the only nonzero entries are `C[1,1]=C[1,0]=C[2,1]=C[2,2]=1`; IoU1=1/3, IoU2=1/2, proposed macro=5/12. Ignored pixel contributes nothing; class 8 is absent. Test false positives for GT-absent classes, all-ignore input and unequal image sizes to distinguish global aggregation from per-image averaging | CPU; policy confirmation before evaluator implementation |
| Geometry and label mapping | Coordinate-coded RGB and mask stay aligned through crop/flips/quarter-turns; official IDs round-trip; nearest-neighbor mask resize creates no new IDs; include one-pixel objects and odd-sized fixtures | Later CPU increment |
| Overlay integrity | Rejected/uncertain records cannot activate; stale hash and duplicate correction fail; activation changes effective mask identity; disabling restores original; raw bytes unchanged | Later CPU increment |
| PNG/ZIP round trip | Synthetic 1024 x 1024 masks covering all IDs survive encode/archive/decode pixel-exactly; wrong modes/sizes/IDs, duplicate/missing/extra names and unexpected paths fail; sorted entries and fixed metadata yield repeatable ZIP bytes in pinned environment | First CPU increment; flat layout is provisional |
| Model/loss/checkpoint | Tiny tensors produce eight spatially aligned logits; ignored pixels have zero loss gradient; all-ignore batch leaves parameters unchanged; finite nonzero valid gradients; checkpoint reload preserves predictions and metadata; epoch-boundary resume reproduces the next update | Later approved numerical environment; synthetic checks only |
| End-to-end baseline | A synthetic checkpoint goes through the real inference, metric and packaging paths; later an authorized execution-machine run demonstrates loss reduction on a tiny official-training subset, then evaluates the frozen holdout and validates every output | CPU plumbing first; optimization only under later execution authorization |

If mask resizing is introduced, use Pillow nearest-neighbor (or explicitly
tested tensor `nearest-exact`) and bilinear interpolation only for continuous
images/features. The first baseline does not resize label masks.
[Interpolation semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html).

## 6. Later execution-machine release

Before any GPU smoke or run, record authorized machine hardware, available VRAM,
RAM, disk space, driver/runtime compatibility, package lock and dataset hashes.
Estimate space for checkpoints, decoded batches, predictions and logs; verify
writable artifact storage separately from read-only raw data. This machine's
capabilities do not establish the execution machine's capacity.

A separately authorized bounded resource check must cover the actual 512 crop
training step (including backward and optimizer state), full 1024 validation/
inference, and checkpoint save/reload. Start with batch one, record peak allocated
and reserved VRAM, host RSS, loader throughput and measured headroom; select a
batch only after the owner reviews those measurements. On OOM or incompatible
software, report and wait; do not silently downscale, change models or install.

For future performance comparisons, fix device, inputs/count, batch, precision,
crop/full-image policy, data loading and checkpoint. Record warm-up and timed
iteration counts, synchronize GPU timing, and report model-only latency separately
from end-to-end decode/predict/PNG time, throughput, peak memory and environment.
Validate equivalent outputs or declared tolerances first. More capable tooling
and newly written code are not evidence of greater speed or accuracy.

Formal training remains owner-started via a foreground command on that execution
machine after approval of the frozen data/split, metric rules, exact config,
finite budget, validation cadence and resource evidence. Provide the command
and stop conditions in that later handoff; do not start it as an Agent action.

## 7. Review decisions and first implementation increment

The owner review now covers the scratch baseline, data-review workflow and the
first CPU increment. Only the following decisions need new evidence:

- Organizer metric edge cases/global aggregation and current reproduction-stage
  notices: resolve before evaluator implementation or experiments as applicable.
- Actual group metadata, review coverage and holdout membership: resolve before
  training; no assertion of scene independence from filenames alone.
- Audit dependency versions/licenses: review before installation. Runtime versions,
  batch, update budget and validation cadence: review with execution-machine facts.
- Final ZIP layout and stage-specific deliverables: resolve before submission.

The first increment implements only read-only PNG/pair auditing, deterministic
manifest/split validation, the proposed submission validator/packager, synthetic
fixtures and usage documentation. Its CLI must never train, download weights,
modify raw data or upload anything. Use explicit input/output arguments resolved
through local configuration; reject outputs inside raw-data locations. Run audit,
identity and PNG/ZIP gates on synthetic fixtures first, then an authorized
read-only audit of official data. This increment does not claim near-duplicate
clearance, human review, model correctness or competition-ready predictions.
Near-duplicate screening and review tooling follow as a separately scoped CPU
increment; model and training code follow only after their own review.

Current design validation: organizer SHA-256, three split hashes/counts,
disjoint/complete split coverage, aggregate pixel count and local inventory
digest match the recorded facts. Relative document links, machine-path exclusion,
the metric example arithmetic and Git whitespace checks passed before committing.
At the design commit, no runtime gate in section 5 had run. The subsequent
CPU increment reports its own evidence separately; it does not release later
model or execution gates or historical model evidence.
