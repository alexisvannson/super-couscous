import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with per-label positive class weights.
    Weights can be provided directly or computed from training data.

    Config keys (under `loss`):
        pos_weight (list[float], optional): Per-label positive weights.
            If omitted, weights are computed from the training DataLoader.
        label_smoothing (float): Smooth targets to [eps, 1-eps]. Default: 0.0.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., 2021).
    Decouples focusing strength for positives and negatives — hard-downweights
    easy negatives while being softer on hard positives.

    Config keys (under `loss`):
        gamma_neg (float): Focusing exponent for negatives. Default: 4.
        gamma_pos (float): Focusing exponent for positives. Default: 1.
        clip (float): Probability margin to shift negatives down. Default: 0.05.
        label_smoothing (float): Smooth targets to [eps, 1-eps]. Default: 0.0.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)

        # Shift negative probabilities down by `clip` to reduce easy-negative dominance
        probs_neg = (probs - self.clip).clamp(min=1e-8)

        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(1 - probs_neg)

        # Per-sample probability used for focal weighting
        pt = targets * probs + (1 - targets) * (1 - probs)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)

        asymmetric_weight = (1 - pt).pow(gamma)
        loss = -(loss_pos + loss_neg) * asymmetric_weight
        return loss.mean()


def compute_pos_weight(train_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Compute per-label positive weights from training data as num_neg / num_pos.
    Clamps to [0.1, 100] to prevent extreme values with very rare labels.
    """
    pos_counts = None
    total = 0

    for _, labels in train_loader:
        labels = labels.float()
        if pos_counts is None:
            pos_counts = labels.sum(dim=0)
        else:
            pos_counts += labels.sum(dim=0)
        total += labels.size(0)

    neg_counts = total - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).clamp(0.1, 100)
    return pos_weight


_LOSS_REGISTRY = {
    "weighted_bce": WeightedBCELoss,
    "asymmetric": AsymmetricLoss,
}


def build_loss(loss_config: dict, train_loader=None) -> nn.Module:
    """
    Instantiate a loss from config.

    Args:
        loss_config: Dict with key `name` and optional hyperparameters.
        train_loader: Required when name=weighted_bce and pos_weight is not set in config.

    Returns:
        nn.Module loss instance.
    """
    name = loss_config.get("name", "asymmetric").lower()

    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSS_REGISTRY)}")

    params = {k: v for k, v in loss_config.items() if k != "name"}

    if name == "weighted_bce":
        if "pos_weight" in params:
            params["pos_weight"] = torch.tensor(params["pos_weight"], dtype=torch.float32)
        elif train_loader is not None:
            print("Computing pos_weight from training data...")
            params["pos_weight"] = compute_pos_weight(train_loader)
            print(f"pos_weight: {params['pos_weight'].tolist()}")
        else:
            print("Warning: weighted_bce without pos_weight — falling back to uniform weights.")

    return _LOSS_REGISTRY[name](**params)
