# Verified Data Facts

Audit date: 2026-09-08. [Summary JSON](summary.json) contains measured counts,
identities and explicitly unperformed checks. No model history is included.

- All 14,514 PNG files decoded: 6,996 training images with matching masks,
  500 Test 1 images, and 11 example image/mask pairs.
- Every PNG is 1024 x 1024. Images are RGB; masks are L. All mask values are
  within 0--8. No damaged PNG or dimension exception was found.
- The reference split contains 6,297 train and 699 validation IDs, with no
  repeated ID, no intersection, and complete coverage of the training pairs.
  The [train](splits/train.txt), [validation](splits/val.txt) and
  [all-training](splits/train_all.txt) files are byte-preserved identity facts,
  not a required split choice for the new implementation.
- No identical decoded training images were found. This does not establish
  absence of near-duplicates, same-scene leakage, train-test overlap or label errors.
- Test 2/3, flight/source grouping and test GT were not available to this audit.
  Example data is not automatically admitted as extra training data.

## Local access

The optional ignored `.local/data-paths.json` defines `dataset_root` and
dataset-relative paths for `train_images`, `train_masks` and `test_images`.
It grants read-only data access, not permission to scan the containing project
for historical code or model decisions. A local detailed content inventory may
be retained beside it; its digest is recorded in the summary.

On another machine, the owner supplies an authorized dataset location and
verifies it against the same inventory before using these facts. No absolute
machine path is required by tracked code or documentation. No symlink, data
copy or import from another project's implementation is part of this setup.
