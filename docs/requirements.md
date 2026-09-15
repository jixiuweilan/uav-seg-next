# Requirements and Open Decisions

Source: the [preserved organizer document](official/无人机低空航拍图像语义分割.md).
This is a local source snapshot, not confirmation of current portal notices.

## Source-derived requirements

- Semantic segmentation of 1024 x 1024 RGB aerial patches using official
  competition data only. Training collection: 6,996 labeled images. The source
  describes Test 1/2/3 collections of 500/1,300/1,794 images.
- Label IDs: 0 Ignore, 1 Background, 2 Building, 3 Road, 4 Water, 5 Barren,
  6 Forest/Vegetation, 7 Agricultural, 8 Vehicle.
- Metric: mean class IoU excluding Ignore. Specify the treatment of predictions
  of 0 at non-ignore GT and absent-class aggregation before implementing the
  evaluator; the snapshot does not fully specify these edge cases.
- Public academic pretrained weights may be fine-tuned. Additional data,
  commercial closed-model APIs, multiple-model ensembles and test-data overlap
  in pretraining are prohibited by the source. Check source and weight licenses
  independently for any selected candidate before use.
- Output: same input filenames, 1024 x 1024 single-channel grayscale PNG,
  IDs 0--8, no RGB or palette PNG, packaged in one ZIP. The snapshot does not
  spell out all ZIP-layout details; resolve ambiguity before submission.
- Later stages require reproducible training/validation code, environment and
  operating instructions, a Docker deliverable and a technical PDF. Verify
  stage-specific notices before an actual submission.

## Owner constraints (not additional organizer quotations)

- One model and one checkpoint; no checkpoint/seed averaging.
- This machine is for code and CPU audits only; training is separately
  authorized and started by the owner on the execution machine.
- Independent foundations are written afresh. The initial fact-only starting
  point remains traceable; the first released implementation is CPU audit/format
  tooling, not a training framework or trained model.
- Keep the independent design small and decision-oriented. Preserve raw data,
  provenance and reproducibility; do not preselect an architecture from history.

## Data-quality capability required in the subsequent design

1. Check corruption, dimensions, pairing and IDs. Version manifests and splits.
2. Screen exact/near duplicates and same-scene or consecutive-frame leakage.
   Keep group provenance and a frozen group-aware validation holdout when
   feasible; do not claim that disjoint filenames guarantee independence.
3. Rank annotation suspicion using prediction/GT disagreement, alignment,
   class confusion, missing-region and boundary signals. Prefer held-out or
   out-of-fold predictions to avoid training-fit bias. Disagreement is a
   review cue, not ground truth or a deletion rule.
4. Review high-risk samples manually, with a sampled low-risk control to assess
   missed issues. Record confirmed versus rejected suspicions. Correct only
   confirmed errors with an annotation tool if needed; preserve original and
   corrected masks, reviewer decisions, hashes and data-version identity.
5. Audit annotation/split changes separately from model changes. Do not clean a
   validation set based on which labels make the candidate score better; freeze
   evaluation labels and keep test information out of the correction decision.

The first approved [CPU increment](cpu-audit.md) implements format, pairing,
identity and reference split checks. The released [scene-review increment](scene-review.md)
adds similarity candidates, review records and confirmed-group checks. Tool
availability is not completed human review or a frozen group-aware holdout.
Prediction-assisted annotation review remains later work. No annotation service
is authorized by this document.

负责人后续要求结束追加复核并继续项目。当前新增的[基线数据准备](baseline-data.md)
交付同步几何变换、标签映射和候选整组划分，均在 CPU 合成数据及已有 JSON 上验证。
候选不等于冻结验证集，不代表指标边界或训练执行已经授权。

2026-09-15继续交付[基线代码与数值验证入口](baseline-runtime.md)。指标以显式命名的
内部口径实现并通过手算测试，尚未确认官方边界等价性；不得用于实验选优。
模型与更新/预测接口已编写，检查点修复后执行端全量5/5通过的v2原始JSON已接收并核对代码摘要；
证据及边界见[跟进记录](checkpoint-fix.md)。后续已交付[有限更新控制](training-control.md)，
正式训练入口及运行授权仍未交付；验证选优当时尚未交付。

0.7.0已交付[验证、确定性调度和合成选择管线](validation-selection.md)。选择器仍限定合成接口，
明确拒绝宣称官方选优；正式训练入口、划分冻结、官方指标确认和运行预算仍未交付。

## Initial implementation acceptance design

Define hand-checkable confusion-matrix/ignore tests; paired geometric transforms
and nearest-neighbor mask resizing; leakage and invalid-data fixtures; exact
submission round-trip tests; deterministic manifest/config/checkpoint identity;
and equivalent-workload speed/memory measurements once a runtime exists.
Choose only enough infrastructure to validate a first end-to-end baseline.
