# CPU Audit and Submission Format Increment

This is the first implementation increment approved after the independent
design. It provides read-only data auditing, deterministic manifests, reference
split checks, and a provisional flat PNG/ZIP validator and packager. It contains
no model, evaluator, training loop, near-duplicate detector or annotation editor.
The owner accepted delivery commit `a464e02` on 2026-09-09.

See the [validation record](data/cpu-audit-report.md) for the completed checks,
the resolved pixel-hash discrepancy and the remaining review boundary.

## Environment

The owner selected a project-local Miniconda environment. From the repository
root, the environment prefix is `.conda/uav-seg-next`, which is ignored by Git.
Use its Python directly; activation is optional. Do not install CUDA or a training
framework on this workstation. The portable direct specifications are in
[environment.yml](../environment.yml). The exact Linux builds and SHA-256 hashes
are pinned in [environment-linux-64.lock](../environment-linux-64.lock), with
package URLs and license metadata in [dependencies.json](dependencies.json).

```bash
conda create --prefix .conda/uav-seg-next --file environment-linux-64.lock
.conda/uav-seg-next/bin/python -m unittest discover -s tests -v
.conda/uav-seg-next/bin/python -m uavseg --help
```

The explicit lock targets Linux x86-64. On another platform, review a fresh solve
of `environment.yml` and validate it before recording a platform-specific lock.
The YAML pins direct versions but does not freeze transitive builds.

No editable installation is needed: execute from the repository root. Python's
standard library supplies the CLI, JSON, hashing, ZIP and test infrastructure.
Pillow decodes/verifies PNGs and NumPy counts mask values. The direct dependency
licenses were checked against first-party documentation: Pillow uses MIT-CMU;
NumPy uses BSD-3-Clause. Preserve the installed packages' copyright/license
notices when redistributing an environment. This increment does not redistribute
those packages or weights.
[Pillow license](https://pillow.readthedocs.io/en/stable/about.html#license),
[NumPy license](https://numpy.org/doc/stable/license.html).

## Data access and audit

The existing ignored `.local/data-paths.json` supplies `dataset_root`,
`train_images`, `train_masks`, `test_images`, and `access: "read-only"`.
Folder references must be portable paths relative to the dataset root and must
remain inside it. A relative dataset root is resolved relative to the config
file. Actual machine paths belong only in ignored local configuration.
The program only reads the three specified folders; it never scans the directory
containing the dataset for source code or model history. Example pairs are not
included in this increment's audit or admitted for training.

```bash
.conda/uav-seg-next/bin/python -m uavseg audit \
  --config .local/data-paths.json \
  --train-split docs/data/splits/train.txt \
  --val-split docs/data/splits/val.txt \
  --all-split docs/data/splits/train_all.txt \
  --output .local/cpu-audit-v1.json
```

Each directory must contain regular, flat `.png` files; empty directories,
unexpected entries and file symlinks fail. Training image/mask filenames must
match exactly. Each PNG is verified and fully decoded, with 1024 x 1024 dimensions,
8-bit RGB images and 8-bit L masks. Masks permit only IDs 0--8; palette, animation
and mask transparency are rejected. The current per-PNG encoded size limit is
32 MiB; exceeding it reports an error for owner review rather than skipping the
sample. Input validation performs no conversion or repair.

The three split files are optional as a set. IDs are basenames without `.png`,
one per line; duplicate, blank, surrounding-whitespace and path-like IDs fail.
If supplied, train/val must be disjoint and their union and train_all must equal
all observed pairs. Decoded-image hashes identify exact duplicate training
groups, including different PNG encodings of identical pixels. A group spanning
train/val fails. This is not a same-scene grouping check or a holdout freeze.

Success writes a version-1 JSON envelope containing:

- `manifest`: sorted training/test records with sample ID, dataset-relative
  image/mask paths, byte count, mode/size, encoded SHA-256, decoded-pixel SHA-256
  and nine-element mask class counts; split IDs and original split-byte hashes;
  exact duplicate groups and explicit unperformed checks.
- `manifest_sha256`: SHA-256 of canonical manifest JSON: UTF-8, sorted keys,
  compact separators, no NaN, one trailing newline. Enumeration order and
  machine location do not enter the content identity. Split source-byte changes
  intentionally change identity even if the ID sets match.
- `producer`: tool version, Python/Pillow/NumPy versions and a digest of canonical
  `{filename: SHA-256}` for the package's Python source files. Producer metadata
  is outside the dataset-content digest.

The version-1 `pixel_sha256` contract hashes **only decoded uint8 pixel bytes**:
rows from top to bottom, pixels left to right, interleaved R/G/B for images or
one label byte per pixel for masks. It includes no mode/size prefix, filename,
PNG metadata, color conversion or orientation transform. Mode and size are
separate required fields; compare them alongside the digest. The encoded-file
`sha256` hashes all original PNG bytes. Equal field names in another inventory
do not establish equal hash definitions; reconcile the definitions before
interpreting different digests as changed data.

The command prints a compact JSON summary to stdout and progress to stderr.
An input or contract error exits with code 2, reports the problem, and publishes
no manifest. It stops at the first failure; the owner decides its resolution.
The tool does not silently exclude a file and continue with a reduced dataset.

## Validate and package predictions

Use the audit's test-image collection as the expected filename set. It refers
to that audited snapshot; a digest check detects accidental manifest modification
but is not proof that data at the original location has remained unchanged.
Re-audit if the official collection changes. Packaging does not read test labels.

```bash
.conda/uav-seg-next/bin/python -m uavseg pack \
  --config .local/data-paths.json \
  --manifest .local/cpu-audit-v1.json \
  --predictions .local/predictions \
  --output .local/submission-v1.zip

.conda/uav-seg-next/bin/python -m uavseg validate-zip \
  --config .local/data-paths.json \
  --manifest .local/cpu-audit-v1.json \
  --archive .local/submission-v1.zip
```

These commands require predictions already produced elsewhere; they do not
generate model predictions. The packager validates PNG bytes, writes sorted
members with fixed ZIP metadata using STORE compression, then reopens and
validates the archive before publishing it. With identical input PNG bytes,
ZIP bytes repeat in the pinned environment. Re-encoding equal pixels can change
the ZIP digest; no stronger cross-encoder byte-identity claim is made.

Archive validation checks every member without extracting it: exact filename
set, no duplicates, directories, nested paths, symlinks, encryption, extra files
or invalid PNGs. It does not assess semantic accuracy, checkpoint provenance or
organizer acceptance. The flat ZIP layout remains provisional until confirmed
by the organizer; there is no upload command.

## Output protection and scope

Both writers reject outputs under the configured raw-data root, inside the
prediction directory, or equal to protected input files. Resolved paths prevent
ordinary symlink aliases from bypassing the root check. Existing output files
are never replaced: choose a new versioned name. Temporary files are removed on
normal failure and final publication is exclusive. This assumes inputs and
directory links remain stable during a command; concurrent mutation is outside
this local audit contract.

Keep manifests, raw/corrected masks, PNGs, ZIPs and installed environment contents
in ignored local storage. Only portable source, tests, dependency specifications, split
references and aggregate audit evidence are versioned. PNGs in tests are generated
in temporary directories and are never copied from competition data.

The tests validate this increment only. Metric edge cases, paired model
transforms, human grouping, correction overlays, numerical model tests,
checkpoint replay and GPU/resource measurements remain later gates. No current
result establishes annotation correctness, scene independence or model quality.
