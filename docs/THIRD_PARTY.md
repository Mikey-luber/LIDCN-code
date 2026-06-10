# Third-Party Components

This repository contains the LID feature extraction and logit calibration code.
Some detector backbones used in experiments depend on upstream projects or
pretrained weights that are not redistributed here.

## Included

- Minimal CNNSpot / ResNet implementation:
  - `networks/resnet.py`
- Gram-Net ResNet implementation:
  - `networks/resnet_gram.py`
- UnivFD helper modules:
  - `networks/univfd_models/`

## Optional External Repositories

Place these repositories under the project root only if you use the
corresponding detector or attack mode.

```text
ForgeLens-main/                 # required for --model_type forgelens
Frequency-Collaborative-main/   # required for --model_type csf
evadingfakedetector-main/       # optional StatAttack support
DBD-main/                       # optional local torchattacks fallback / experiments
UniversalFakeDetect-main/       # optional UnivFD weight/layout compatibility
```

The scripts intentionally do not vendor these projects. Please follow each
upstream project's license and installation instructions.

## Weights

Pretrained detector checkpoints and trained calibrator checkpoints are not
included. Pass them explicitly, for example:

```bash
python extract_lid_features.py --model_type resnet --model_path /path/to/cnnspot.pth ...
python train_logit_calibrator.py --load_model_path /path/to/calibrator.pt ...
```

Do not commit large binary weights to the GitHub repository.
