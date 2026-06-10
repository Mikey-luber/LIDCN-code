# Extracted NPZ Format

`extract_lid_features.py` writes one `.npz` file per extraction run. This file
is the stable interface between detector feature extraction and calibrator
training/evaluation.

Important fields:

- `features`: stacked LID features.
  - clean samples are at `features[:n_clean]`
  - adversarial samples start at `features[n_neg_samples:n_neg_samples+n_adv]`
- `labels`: binary LID detector labels, `0=clean/noisy`, `1=adversarial`
- `n_neg_samples`: number of negative samples in `features`
- `clean_model_logits`: raw detector logits for clean samples
- `adv_model_logits`: raw detector logits for adversarial samples
- `clean_true_labels`: original image labels for clean samples
- `clean_effective_labels`: labels after any detector convention flip
- `adv_true_labels`: original image labels for adversarial samples
- `attack_success_mask`: whether the base detector was fooled
- `attack_valid_mask`: whether the clean sample was initially classified correctly
- `layers`: layer names used for LID features
- `model_type`, `attack_type`, `epsilon`, `lid_k`, `feat_norm`,
  `lid_distance_metric`, `ref_mode`, `batch_ref_size`: provenance metadata

The calibrator consumes:

```text
LID vector + raw detector logit
```

when trained with `--use_logits_input`.

Always keep feature extraction settings fixed between training and evaluation.
Changing `--include_input`, `--layers`, `--feat_norm`, `--lid_distance_metric`,
`--lid_k`, or `--ref_mode` changes the calibrator input distribution.
