# First CPU Increment: Validation Record

Audit and tests: 2026-09-08. Delivery review: 2026-09-09.
Scope: the first CPU audit/format increment approved after design commit
`9d7dc61`. [Machine-readable evidence](cpu-audit-summary.json) records content,
source, test and environment identities. [Usage and contracts](../cpu-audit.md)
describe the implemented commands.

Owner acceptance: passed on 2026-09-09 for implementation commit `a464e02`.
The owner reviewed the delivered evidence and explicitly accepted this phase.
This closes the first CPU increment; subsequent capability releases remain
separate decisions.

## Verified results

| Check | Observed result |
| --- | --- |
| Synthetic contract tests | 19 tests passed, including CLI audit/pack/validate round trips, invalid-data cases, split errors, output protection and pixel-hash definition |
| Official training pairs | 6,996 images and 6,996 masks fully decoded and validated |
| Test 1 images | 500 images fully decoded and validated |
| Dimensions, modes and IDs | All 14,492 scoped PNGs met the 1024 x 1024 RGB/L contracts; every mask value was 0--8 |
| Reference splits | 6,297 train / 699 validation; no repeated IDs or intersection, and complete coverage of all 6,996 pairs |
| Exact decoded training duplicates | No groups found |
| Original file identity | All 14,492 file byte hashes, byte counts, modes and sizes matched the supplied inventory |
| Pixel identity reconciliation | Both documented hash definitions reproduced their recorded values on every scoped file |
| Environment | Project-local Miniconda; Python 3.12.14, Pillow 12.3.0, NumPy 2.5.3; package consistency check passed |

The 11 example pairs are outside this increment's scope. This explains the
difference from the starting audit's count of 14,514 PNGs; no sample was removed.
Aggregate training-label counts and all three split source hashes also match
the original [fact summary](summary.json). Raw data was only read.

## Pixel-hash discrepancy and resolution

The initial comparison incorrectly assumed that identically named
`pixel_sha256` fields used the same definition. The first differing digest
belonged to `train/images/0000.png`, whose encoded-file SHA-256 was unchanged.
Data-only checks identified a metadata prefix in the supplied inventory:

```text
Current schema 1: SHA256(pixel_bytes)
Supplied inventory: SHA256(UTF8(mode) + UTF8(str((width, height))) + pixel_bytes)
```

For example, the reference prefix for an RGB image is `RGB(1024, 1024)` without
a trailing separator. Pixel bytes in both checks are decoded uint8 values in
row-major order, with interleaved RGB channels for images and one byte per mask
pixel. Re-decoding all 14,492 scoped files reproduced both sets of digests and
confirmed their encoded-file hashes again. The difference is a hash convention;
it is not evidence of changed data or differing decoded pixels.

No historical implementation was inspected. The independent audit retains its
own documented schema-1 definition. A synthetic regression test fixes that
definition using known pixel bytes. The supplied inventory remains unchanged.

The complete manifest and one-off reconciliation script/result remain in ignored
`.local/` storage. Their identities are retained in the evidence JSON; no image,
mask, archive, installed environment or machine path is committed.

## Remaining review boundary

These results accept the CPU data/format capability only. They do not establish
near-duplicate or scene independence, annotation correctness, train-test overlap
clearance, model accuracy, training reproducibility or execution-machine capacity.
The provisional flat ZIP layout still needs organizer confirmation before a real
submission. Packaging was exercised on synthetic predictions; no model prediction
or competition submission was produced.

Near-duplicate screening and human-review tooling are the next separately scoped
CPU increment. Model implementation, correction activation, GPU checks and formal
training remain unreleased. This delivery creates no remote and starts no run.
