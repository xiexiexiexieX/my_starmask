# MAINet v4 Notes

## Purpose

`mainet/v4` is the experimental successor to `mainet/v3`.

`v3` is kept as the original zero-annotation morphology backbone:

- learnable Moffat PSF channel
- fixed multi-orientation strip channel
- SKFusion selective fusion
- FPN/RPN/ROI with strides `(2, 4, 8, 16)`

`v4` contains the new AP75/MSE-oriented changes:

- PSF becomes `PSFGatedEnhancement`, a gated residual enhancement branch.
- Fixed 4-direction strip sampling is replaced by deformable orientation strip sampling.
- SK-style competition is replaced by `GatedResidualFusion`.
- FPN/RPN uses memory-aware fixed strides `(2, 4, 8, 16)`.
- RPN anchors include elongated ratios `(0.25, 0.5, 1.0, 2.0, 4.0)`.
- RPN head uses GroupNorm.
- Mask loss is `BCE + Dice`.
- Mask head uses coarse prediction plus residual refinement.

## Key Design

Pipeline:

```text
Input
  -> Stem
  -> PSF gated residual enhancement
  -> Deformable Orientation Strip
  -> Gated residual fusion
  -> FPN
  -> RPN with elongated anchors
  -> ROIHead with BCE + Dice mask loss
```

The main idea is to make morphology priors act as residual suggestions instead of
overwriting the backbone representation. This should reduce feature collapse,
improve streak continuity, and make ablations less brittle.

## Important Files

- `backbone.py`: PSFGatedEnhancement, DeformableOrientationStrip, GatedResidualFusion.
- `deform_strip/__init__.py`: module export for the new strip branch.
- `heads.py`: RPN anchors/GN and mask BCE+Dice/refinement branch.
- `model.py`: v4 strides and FPN channels.
- `train.py`: v4 output directory and default training config.
- `ablation.py`: v4 ablation training only. Evaluation is handled by `eval.py`.

## Training

From project root:

```bash
python mainet/v4/train.py --debug --epochs 1
python mainet/v4/train.py --epochs 100
```

Main-model checkpoints are saved to:

```text
work_dirs/mainet/v4/best_model.pt
```

## Ablation Training

From project root:

```bash
python mainet/v4/ablation.py --list
python mainet/v4/ablation.py --ablation strip --debug --epochs 1
python mainet/v4/ablation.py --ablation strip --epochs 100
python mainet/v4/ablation.py --all --epochs 100
```

Ablation checkpoints are saved below the v4 folder:

```text
work_dirs/mainet/v4/<ablation>/best_model.pt
```

For example:

```text
work_dirs/mainet/v4/strip/best_model.pt
work_dirs/mainet/v4/psf/best_model.pt
work_dirs/mainet/v4/full/best_model.pt
```

The `plain` ablation branch is an identity branch, not a convolutional branch.
This keeps ablation factors clean:

```text
none   = Identity + Identity + Add
psf    = PSF-gate + Identity + Add
strip  = Identity + DeformStrip + Add
fusion = Identity + Identity + ResidualFusion
full   = PSF-gate + DeformStrip + ResidualFusion
```

## Evaluation

Evaluation is centralized in `eval.py`.

```bash
python eval.py
python eval.py --model v4
python eval.py --model v4 --ablation strip
python eval.py --model v4 --ablation all
python eval.py --all
```

Command meanings:

- `python eval.py`: evaluate only the main v4 checkpoint.
- `python eval.py --model v4 --ablation strip`: evaluate one v4 ablation.
- `python eval.py --model v4 --ablation all`: evaluate all v4 ablations.
- `python eval.py --all`: scan and evaluate every discovered checkpoint under `work_dirs/`.

## Notes

- `eval.py` defaults to the main v4 model. Full checkpoint discovery requires `--all`.
- `mainet/v4/ablation.py --eval-only` is kept only as a compatibility hint; use `eval.py` for actual ablation evaluation.
- Stride-1 FPN/RPN was removed from v4 because 512x512 stem features made
  anchor generation and FPN activations too expensive. The RPN is fixed in
  ablation experiments and is not treated as a variable factor.
- Existing v3 checkpoints should continue to use `mainet/v3`.
