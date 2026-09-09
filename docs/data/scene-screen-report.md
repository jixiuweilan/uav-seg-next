# Scene Screening Increment: Initial Evidence

Date: 2026-09-09. Scope: the CPU scene screening and review work released by
the owner after acceptance of the first increment. See the
[tool contract](../scene-review.md) and [measured summary](scene-screen-summary.json).
This records screening results and working review tools, not human-confirmed
scene relationships or a new validation split.

## Observed screening results

| Item | Result |
| --- | --- |
| Audited training images reverified | 6,996; every encoded-file hash matched the input manifest |
| Unordered pairs screened | 24,468,510, using the recorded eight-pose hash/RMS method |
| Qualifying cross-split cues | 99; all 99 retained in the first review queue |
| Qualifying within-split cues | 592; the strongest 100 retained, 492 explicitly omitted from this batch |
| Random non-qualifying controls | 20, seed 0 |
| Low-texture images flagged | 1; no sample deleted |
| First offline review page | 219 pairs referencing 179 distinct original images |
| Human decisions / confirmed groups | 0 / 0; status is `review_pending` |

The 99 cross-split cues are possible shared-scene/duplicate relationships, not
99 established leakage cases. Similarly, zero confirmed groups currently means
no human decisions have been entered, not absence of scene overlap. Thresholds
remained at the predeclared defaults throughout this run. Test images and masks
were not read by screening; the audited manifest supplies only their prior
identity context. Reference split bytes are unchanged.

The review cap is a batch limit. It does not dismiss the remaining 492 qualifying
pairs. A later batch can use a larger `--max-per-partition` and a new output name;
regenerated candidates have a new screen identity and review records must be
explicitly reconciled before reuse. Unscreened flight/source relationships may
also be entered as manual relations with human evidence. Thumbnail screening
does not establish recall for overlapping crops or consecutive frames.

## Validation

All 31 unittest tests passed: 19 retained CPU contract tests and 12 scene/review
tests. The latter include transformed/brightness-adjusted synthetic pairs,
different encodings of exact pixels, unrelated and low-texture images, fixed
queue/control sampling, stale data, required decision provenance, transitive
groups, cross-split conflicts, contradictory rejections and protected outputs.
The Node harness executed the generated JavaScript export/resume path on a
synthetic page and checked that unchanged imported decisions retain provenance.

The actual 219-pair page was parsed to verify its pair count, local original-image
links, embedded preview sources and matching screen identity. Its JavaScript
passed syntax checking. Browser layout and opening original-image links were
not exercised by an automated browser; open the HTML locally on the data machine.
No dependency was added. The existing environment consistency check passed.

Source, test, environment, screen, group-report and review-page digests are
retained in the measured summary. The accepted first audit remains tied to its
original source version; its producer record was not rewritten for this increment.

## Local handoff

The generated artifacts are in ignored local storage:

- `.local/scene-candidates-v1.json`: retained candidates, controls, parameters and counts.
- `.local/scene-review-v1.html`: open locally, compare originals, record and download decisions.
- `.local/scene-groups-pending-v1.json`: the current pending queue and ungrouped IDs.

No real decision was prefilled, no scene group was asserted and no annotation
or split was changed. The next data-dependent action is human review of the
queued relations, starting with the cross-split candidates, followed by checking
the exported decisions with `check-groups`. Model code, training, GPU checks and
test-data overlap clearance are outside this delivery.
