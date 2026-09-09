# Training Scene Screening and Human Review

The owner released this next CPU increment after accepting `a464e02`. Continue
within the approved design without asking the owner to redefine routine details.
This increment supplies image-similarity cues, an offline review page and
confirmed-relation group checks. It does not certify scene independence, edit
labels, choose a new split, or train a model.

The [initial evidence record](data/scene-screen-report.md) reports the first
official-training scan and the generated review batch.

## Screen the audited training collection

Use the existing Miniconda environment and the accepted audit manifest:

```bash
.conda/uav-seg-next/bin/python -m uavseg screen-scenes \
  --config .local/data-paths.json \
  --manifest .local/cpu-audit-v1.json \
  --output .local/scene-candidates-v1.json
```

Only training images are read. Their filename set, dataset-relative paths and
encoded-file SHA-256 values must match the audited snapshot before they supply
descriptors. Test images and all masks are excluded from feature extraction.
Any stale image fails the command instead of silently changing the dataset.

The version-1 method is an explicitly defined starting heuristic, not a learned
detector or a claimed complete scene-recognition method:

1. Convert RGB to Pillow L, then resize to 32 x 32 using LANCZOS.
2. For each of eight poses, average each 4 x 4 cell into an 8 x 8 grid. Encode
   `cell > grid mean` as 64 bits in row-major, most-significant-bit-first order.
   Poses are four counterclockwise quarter turns, then horizontal reflection
   followed by those same four turns. The stored pose transforms the right image
   to compare against the left image.
3. For each unordered training pair, compare the left identity hash with all
   right poses. Retain poses with Hamming distance at most 8. Compare their
   32 x 32 gray arrays after subtracting each image's mean and dividing by
   `max(std, 1)`. An eligible pose with normalized RMS distance at most 0.45 is
   a candidate. Distance reductions use float64; descriptors use float32.
4. Images with grayscale standard deviation below 5 are flagged as low texture.
   Similarity alone does not shortlist pairs involving them; equal audited
   decoded-pixel hashes still qualify as exact matches. Flagged images remain
   in the dataset and require source evidence/manual attention, not deletion.

The thresholds are declared implementation defaults, not validated operating
points or results of model/test-set tuning. Simple hashes can miss partial crop
overlap, viewpoint changes and consecutive frames, or confuse repeated textures.
Unrelated scenes can look alike; shared flight/source metadata is still needed
when available. The test collection never calibrates these thresholds.

All pairs are screened. To bound review material, retain at most 100 qualifying
pairs per partition: cross train/val, within a split, or unassigned when the audit
has no split. Exact matches rank first, then lower RMS and Hamming distance,
then stable IDs. Counts of every qualifying and omitted pair remain in the output.
The review page displays retained cross-split candidates first.

The default control sample requests 20 distinct pairs from the non-qualifying
remainder, using Python's seeded RNG with seed 0 and rejection sampling. It does
not accidentally draw truncated qualifying pairs. The loop is bounded; report
the actual retained count when there are too few eligible controls or the bound
is reached. Controls include difficult/unscored low-texture cases when sampled.
They help inspect missed relations; 20 pairs cannot establish a recall estimate
for all scenes. `--hash-max`, `--rms-max`, `--min-std`, `--max-per-partition`,
`--controls` and `--seed` are recorded parameters; any change creates a separate
candidate version and requires compatible review records.

The envelope contains the canonical `screen` payload, its `screen_sha256`, and
source/environment provenance. It binds the input manifest identity. Pair IDs
hash canonical `[left_id, right_id]`, with distinct IDs in sorted order. Stored
RMS values are rounded to eight decimal places for reporting; thresholds and
ranking use the unrounded computed value. A candidate is never an accepted edge.

## Open and complete the offline review

```bash
.conda/uav-seg-next/bin/python -m uavseg review-scenes \
  --config .local/data-paths.json \
  --manifest .local/cpu-audit-v1.json \
  --screen .local/scene-candidates-v1.json \
  --output .local/scene-review-v1.html
```

Open the generated HTML as a local file on the machine holding the dataset.
It embeds 256-pixel previews and links to full-resolution original images. The
best aligned right preview is available in a details panel. Original links use
local file URIs, so this artifact is restricted to `.local/`, is not portable to
another machine unchanged and must not be published. Regenerate it there using
that machine's authorized data config. Rendering verifies every linked image's
byte identity again. The page has no external scripts, fonts, uploads or service.

For each relation, compare original images or source metadata, enter the reviewer
identity, evidence type and reason, and choose:

- **Confirmed**: the pair shares a scene or duplicates image content.
- **Rejected**: the pair represents different scenes; this is stronger than
  merely saying it is not an exact duplicate.
- **Uncertain**: the evidence does not support a definite conclusion.
- **Unreviewed**: leave pending; it is not exported as a completed decision.

Download the decisions JSON and keep each revision as a new local file. The page
does not save automatically or write to the dataset. Use its resume input to load
an earlier download. Unchanged imported decisions preserve their original
reviewer/timestamp; changed decisions receive the entered reviewer and a new ISO
timestamp. The CLI validates exported records before any grouping. Reviewer
identity and evidence are declarations, not authenticated signatures.

The document contains `schema_version: 1`, `kind: "scene-decisions"`, the exact
manifest/screen digests and a `decisions` list. Each decision contains `pair_id`,
`left`, `right`, `status`, `origin`, `reviewer`, timezone-aware `reviewed_at`,
`reason` and `evidence`. Evidence is `full_resolution_images` or `source_metadata`.
Reasons should identify the actual shared/different landmarks or source record.
For queued pairs use `origin: "screen"`.

A reviewer can also append a relation absent from the queue using known training
IDs and `origin: "manual"`, with the same required evidence and provenance.
This supports independently supplied source/flight relationships without making
thumbnail detection a prerequisite. Recompute its pair ID from the sorted IDs;
the HTML resume/export preserves such manual records. It does not provide a
metadata import service or infer flight IDs from filenames.

## Check confirmed groups and the current split

```bash
.conda/uav-seg-next/bin/python -m uavseg check-groups \
  --config .local/data-paths.json \
  --manifest .local/cpu-audit-v1.json \
  --screen .local/scene-candidates-v1.json \
  --decisions .local/scene-decisions-v1.json \
  --output .local/scene-groups-v1.json
```

Omit `--decisions` to report the untouched pending queue; this does not fabricate
reviews. All referenced IDs and manifest/screen identities must match. Duplicate
decisions, invalid statuses, missing reasons/reviewers/evidence and timestamps
without timezones fail. Only confirmed relations join connected components.
Rejected and uncertain relations never create edges. Groups retain their
confirmed-edge IDs; the report binds the complete decisions digest. Members with
no confirmed connection remain explicitly ungrouped, not presumed independent.

A confirmed component spanning train and validation produces `known_split_conflict`.
A rejected relation inside a confirmed component produces `inconsistent_review`.
Both generate an actionable report and exit 2; unlike malformed inputs, these
are valid reports describing unresolved data/review conflicts. Neither rewrites
the reference split. All other successfully generated reports exit 0:

- `review_pending`: queued relations are unreviewed/uncertain, candidates were
  omitted by the cap, or a component exceeds 20 members and needs a chain review.
- `reviewed_cues_only`: the retained cues have decisions; scene independence is
  still not established. Exit 0 is not a training release.

Group IDs hash the manifest identity and sorted members. The report includes
per-queue decision counts, unreviewed/uncertain IDs, omitted candidate counts,
large-component flags and whether an audited split was available. Every report
keeps `scene_independence_certified: false`. No thresholds, review completion
fraction or lack of detected conflicts automatically freezes an evaluation split.

## Validation scope

Run the repository's unittest discovery command. Synthetic fixtures cover
rotation/reflection, brightness changes, exact pixels with different encoding,
low texture, deterministic queue caps/controls, stale-image rejection, confirmed
cross-split and transitive conflicts, decision provenance, protected outputs and
the CLI flow. If Node is available, `tests/review_dom.cjs` also executes the actual
generated JavaScript's export/resume logic with a minimal DOM. This is not a
browser layout or full-resolution link-opening test; that visual check remains
part of using the offline artifact. Node is not a runtime dependency and is not
installed by the Conda environment.

The accepted first audit is a historical snapshot tied to its own source commit.
The new tools consume its content identity; they do not rewrite its recorded
producer hash to match a later implementation. Raw data, reference splits,
annotation masks and previous audit artifacts remain unchanged.
