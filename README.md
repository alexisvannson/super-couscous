# CheXNet Reimplementation

Reimplementation of **CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning** (Rajpurkar et al., 2017) — [arxiv.org/pdf/1711.05225](https://arxiv.org/pdf/1711.05225).

The paper trains DenseNet-121 on the NIH ChestX-ray14 dataset for multi-label classification of 14 pathologies and demonstrates radiologist-level performance on pneumonia detection. This reimplementation reproduces the core pipeline while documenting every design decision and the deviations made.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view chest X-rays from 30,805 unique patients, annotated with 14 pathology labels extracted via NLP from radiology reports. Labels are noisy (NLP-derived, not radiologist-verified) and severely imbalanced (e.g. Hernia ~0.2%, Infiltration ~18%, No Finding ~53%).

**Paper split:** random 70/10/20 train/val/test with no patient overlap.

**This implementation:** patient-level stratified split (70/20/10 train/val/test by default). Splitting is done at the patient ID level before any index assignment — a patient's images appear in exactly one split. This is stricter than a naive random image split and prevents data leakage since many patients have multiple scans.

---

## Model

**DenseNet-121** implemented from scratch with layer names mirroring torchvision's implementation to allow direct weight loading without key remapping.

Architecture:
- Stem: 7×7 conv, stride 2 → BN → ReLU → MaxPool → 64 channels, 56×56
- 4 Dense Blocks with (6, 12, 24, 16) layers and growth rate k=32
- 3 Transition Layers with compression θ=0.5
- Final BN → ReLU → Global Average Pooling → Linear(1024, num_classes)

**Deviation from paper:** the paper replaces the final FC layer with a single sigmoid output for pneumonia only. This implementation uses a multi-label head with `num_classes=14` outputs followed by independent sigmoid activations, enabling simultaneous classification of all 14 pathologies.

**Pretrained weights:** ImageNet-pretrained backbone loaded from torchvision (`DenseNet121_Weights.IMAGENET1K_V1`). The classifier head is re-initialised from scratch with Kaiming normal initialisation. This matches the paper.

**Backbone freezing:** optionally freeze the backbone to train the classifier head only (`fine_tune.freeze_backbone: true` in config). Not done in the paper but useful for quick baselines.

---

## Loss Function

**Paper:** weighted binary cross-entropy with class-frequency weights:

```
L(X, y) = −w₊ · y log p(Y=1|X) − w₋ · (1−y) log p(Y=0|X)
w₊ = |N| / (|P| + |N|),   w₋ = |P| / (|P| + |N|)
```

**This implementation offers two options** (switchable via `loss.name` in config):

**`weighted_bce`** — faithful reproduction of the paper's loss. Per-label positive weights computed from training data as `num_neg / num_pos`, clamped to [0.1, 100] to prevent instability on extremely rare labels. Matches the paper.

**`asymmetric` (default)** — Asymmetric Loss (Ridnik et al., 2021). Decouples the focal weighting between positives and negatives: easy negatives are down-weighted aggressively (`gamma_neg=4`) while hard positives are treated more gently (`gamma_pos=1`). A probability margin clip (`clip=0.05`) shifts negative probabilities down before computing the log, further reducing the contribution of trivially-negative samples. This is a deliberate improvement over the paper's loss, motivated by the severe label imbalance in NIH CXR14 where the model sees far more negatives than positives for every class.

Optional `label_smoothing` (default 0.0, suggested 0.1) is available for both losses to handle NLP-label noise.

---

## Training

**Paper:** Adam (β₁=0.9, β₂=0.999), lr=0.001, batch size 16, ReduceLROnPlateau (factor 10× on val loss plateau), model selected by lowest validation loss.

**This implementation** matches these defaults exactly. Configurable via `configs/config.yaml` and per-model overrides in `configs/<model>.yaml`. Supported optimisers: Adam, AdamW, SGD with Nesterov momentum.

Early stopping is applied on validation macro F1 (rather than validation loss) with configurable patience. The best checkpoint is saved each time macro F1 improves.

After training, per-label decision thresholds are tuned on the validation set and saved as `<model>_thresholds.npy` (see Evaluation).

---

## Data Augmentation

**Paper:** resize to 224×224, ImageNet normalisation, random horizontal flip only.

**This implementation** extends augmentation while staying conservative — aggressive augmentation can destroy clinically meaningful features in X-rays:

| Transform | Value | Rationale |
|---|---|---|
| Resize | 224×224 | Matches paper |
| RandomHorizontalFlip | — | Matches paper. Lungs are roughly symmetric |
| RandomRotation | ±10° | Patients are not always perfectly aligned |
| RandomAffine (translate) | ±5% | Positional variation across scanners |
| ColorJitter (brightness, contrast) | 0.2 | Simulates different scanner exposures |
| Normalise | ImageNet mean/std | Matches paper |

Augmentation is applied **only to the training set**. Val and test sets receive only resize + normalise. This is enforced via `TransformSubset` which wraps each split with its own transform after splitting, so augmentation never leaks into evaluation.

Augmentation is **not** applied by default in the paper's spirit for the val/test sets.

---

## Evaluation

**Paper:** F1 score with 95% bootstrap confidence intervals on pneumonia only, compared against radiologist performance.

**This implementation** evaluates all 14 labels:

**Macro F1** — average F1 across all 14 labels at a decision threshold. Threshold-dependent and sensitive to class imbalance. Matches the paper's evaluation methodology. Per-label and macro F1 with 95% bootstrap CIs are reported.

**Per-label threshold tuning** — rather than a fixed threshold of 0.5, optimal per-label thresholds are found by sweeping [0.10, 0.90] on the validation set and selecting the threshold maximising per-label F1. Thresholds are always tuned on val and applied to test — never tuned on test. This improves operational performance substantially for rare classes where the model's calibrated probabilities rarely exceed 0.5.

**AUC-ROC** — computed during training (logged to W&B) but not yet in the standalone `evaluate.py` test report. AUC is threshold-independent and is the primary metric for comparing models against NIH CXR14 benchmarks. F1 reflects deployment performance; AUC reflects discrimination ability.

Bootstrap CIs follow the percentile method (1000 iterations, 95% CI) matching Rajpurkar et al.

---

## Class Activation Maps

CAMs are implemented following Zhou et al. (2016) and used in the paper to localise pathologies:

```
Mc = Σk  wc,k · fk
```

where `fk` is the k-th feature map from the final convolutional block (7×7 for 224px input) and `wc,k` is the classifier weight for class `c`. A forward hook captures features before Global Average Pooling without modifying the model. Maps are bilinearly upsampled to input resolution and overlaid as a heatmap.

```bash
python scripts/cam.py densenet \
    --weights models/checkpoints/best.pth \
    --image data/sample/images/00000013_005.png \
    --labels Pneumonia Infiltration \
    --out cam_output/
```

---

## Project Structure

```
configs/
  config.yaml           # base config (data paths, loss, dataloader, transforms, model registry)
  <model>.yaml          # per-model overrides (training, model_params, fine_tune)
models/
  DenseNet.py           # DenseNet-121 from scratch
  CNN.py                # lightweight CNN baseline
scripts/
  thedataloader.py      # ChestXrayDataset, patient-level splitting, TransformSubset
  training.py           # training loop, transforms, threshold tuning, W&B logging
  evaluate.py           # test-set evaluation with bootstrap CIs
  losses.py             # WeightedBCELoss, AsymmetricLoss, build_loss
  cam.py                # Class Activation Map generation and visualisation
tests/
  test_dataloader.py
  test_densenet.py
  test_cnn.py
```

---

## Usage

**Train:**
```bash
python scripts/training.py densenet
python scripts/training.py densenet --weights models/checkpoints/densenet_epoch10.pth  # resume
```

**Evaluate on test set:**
```bash
# with tuned thresholds (recommended)
python scripts/evaluate.py densenet --weights models/checkpoints/best.pth --tune

# with saved thresholds from training
python scripts/evaluate.py densenet --weights models/checkpoints/best.pth \
    --thresholds models/checkpoints/densenet_thresholds.npy
```

---

## Reference

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Ball, R. L., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. https://arxiv.org/pdf/1711.05225
