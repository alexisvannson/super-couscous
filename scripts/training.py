import argparse
import os
import sys
import time
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm


# Add the project root directory to the Python module search path to enable imports from sibling directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.thedataloader import get_dataloaders
from scripts.losses import build_loss


def load_config(model_name):
    """Load base config, optionally merged with model-specific config."""
    base_path = os.path.join("configs", "config.yaml")
    try:
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {base_path}")

    model_path = os.path.join("configs", f"{model_name.lower()}.yaml")
    if os.path.exists(model_path):
        with open(model_path) as f:
            config.update(yaml.safe_load(f) or {})

    return config


def create_model(model_name, config):
    """Create model instance based on name and config using registry."""
    registry = config.get("model_registry", {})
    model_key = model_name.lower()

    if model_key not in registry:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(registry)}")

    entry = registry[model_key]
    module_path, class_name = entry["module_path"], entry["class_name"]

    module = __import__(module_path, fromlist=[class_name])
    ModelClass = getattr(module, class_name)

    return ModelClass(**config["model_params"])


def get_transforms(config):
    """Eval/val/test transform — no augmentation."""
    size = config.get("image_size", 224)
    t = [transforms.Resize((size, size)), transforms.ToTensor()]
    if config.get("normalize"):
        t.append(transforms.Normalize(
            mean=config.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=config.get("normalize_std",  [0.229, 0.224, 0.225]),
        ))
    return transforms.Compose(t)


def get_train_transform(config):
    """Train transform — includes augmentations from config."""
    size = config.get("image_size", 224)
    aug  = config.get("augmentation", {})
    t = [transforms.Resize((size, size))]
    if aug.get("horizontal_flip"):
        t.append(transforms.RandomHorizontalFlip())
    if aug.get("rotation_degrees", 0):
        t.append(transforms.RandomRotation(aug["rotation_degrees"]))
    if aug.get("translate"):
        t.append(transforms.RandomAffine(degrees=0, translate=tuple(aug["translate"])))
    if aug.get("brightness_jitter", 0) or aug.get("contrast_jitter", 0):
        t.append(transforms.ColorJitter(
            brightness=aug.get("brightness_jitter", 0),
            contrast=aug.get("contrast_jitter", 0),
        ))
    t.append(transforms.ToTensor())
    if config.get("normalize"):
        t.append(transforms.Normalize(
            mean=config.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=config.get("normalize_std",  [0.229, 0.224, 0.225]),
        ))
    return transforms.Compose(t)


def tune_thresholds(model, val_loader, device):
    """Find per-label thresholds maximising F1 on the validation set."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.float().numpy())

    probs   = np.concatenate(all_probs)    # (N, C)
    targets = np.concatenate(all_targets)  # (N, C)

    best_thresholds = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            preds = (probs[:, c] >= t).astype(float)
            tp = (preds * targets[:, c]).sum()
            fp = (preds * (1 - targets[:, c])).sum()
            fn = ((1 - preds) * targets[:, c]).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[c] = best_t
    return best_thresholds


def validate(model, val_loader, criterion, device, threshold=0.5, thresholds=None):
    """
    Validate the model on validation set.

    Returns:
        tuple: (avg_loss, exact_match_accuracy, macro_f1, mean_auc, per_label_auc)
            - exact_match_accuracy: % of samples where ALL labels are correct
            - macro_f1: average F1 across all labels
            - mean_auc: average AUC-ROC across all labels
            - per_label_auc: dict {label_index: auc_value}
    """
    model.eval()
    val_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for sample, label in val_loader:
            sample = sample.to(device)
            label = label.to(device).float()

            logits = model(sample)
            val_loss += criterion(logits, label).item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(label.cpu())

    avg_loss = val_loss / len(val_loader)

    probs_cat = torch.cat(all_probs).numpy()   # (N, C)
    targets_cat = torch.cat(all_targets)       # (N, C)

    # Apply per-label or scalar threshold
    t = thresholds if thresholds is not None else threshold
    preds_cat = torch.tensor((probs_cat >= t).astype(float))  # (N, C)
    targets_np = targets_cat.numpy()

    # Exact match accuracy (all labels must match)
    exact_match = (preds_cat == targets_cat).all(dim=1).float().mean().item() * 100

    # Macro F1 across labels
    tp = (preds_cat * targets_cat).sum(dim=0)
    fp = (preds_cat * (1 - targets_cat)).sum(dim=0)
    fn = ((1 - preds_cat) * targets_cat).sum(dim=0)
    macro_f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item() * 100

    # Per-label AUC-ROC (skip labels with only one class present in val set)
    per_label_auc = {}
    auc_values = []
    for i in range(probs_cat.shape[1]):
        if len(set(targets_np[:, i])) < 2:
            continue
        auc = roc_auc_score(targets_np[:, i], probs_cat[:, i])
        per_label_auc[i] = auc
        auc_values.append(auc)
    mean_auc = float(np.mean(auc_values)) * 100 if auc_values else 0.0

    return avg_loss, exact_match, macro_f1, mean_auc, per_label_auc


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    val_loader=None,
    patience=5,
    scheduler=None,
    output_path="weights",
    weights_name="final_model",
    start_weights=None,
    use_wandb=False,
    label_names=None,
):
    best_score = - float("inf")
    patience_counter = 0

    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device))

    os.makedirs(output_path, exist_ok=True)
    print(f"Training model in {output_path}")

    start_time = datetime.now()
    log_path = os.path.join(output_path, f"training_logs_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w") as f:
        f.write(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {epochs}, Patience: {patience}\n")
        f.write(f"Output path: {output_path}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")

    print("start training")

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0.0
        model.train()

        for sample, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0

        monitor_loss = avg_train_loss
        macro_f1 = 0.0

        if val_loader is not None:
            val_loss, exact_match, macro_f1, mean_auc, per_label_auc = validate(model, val_loader, criterion, device)
            monitor_loss = val_loss
            if scheduler is not None:
                scheduler.step(val_loss)
            log_line = (f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f},"
                        f" val_loss={val_loss:.4f}, exact_match={exact_match:.2f}%,"
                        f" macro_f1={macro_f1:.2f}%, mean_auc={mean_auc:.2f}%")

            if use_wandb:
                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "exact_match": exact_match,
                    "macro_f1": macro_f1,
                    "mean_auc": mean_auc,
                }
                for i, auc in per_label_auc.items():
                    name = label_names[i] if label_names and i < len(label_names) else f"label_{i}"
                    metrics[f"auc/{name}"] = auc * 100
                wandb.log(metrics)
        else:
            log_line = f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}"
            if use_wandb:
                wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
            f.write(f"Epoch {epoch+1}/{epochs}, needed {epoch_time / 60:.2f} minutes\n")

        if macro_f1 > best_score:
            best_score = macro_f1
            patience_counter = 0
            best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (monitor_loss={monitor_loss:.4f}): {best_model_path}")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    final_model_path = os.path.join(output_path, f"{weights_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    with open(log_path, "a") as f:
        f.write("-" * 50 + "\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best macro_f1 achieved: {best_score:.4f}\n")
        f.write(f"Final model saved: {final_model_path}\n")


def resolve_path(paths, writable=False):
    """Return the first valid path from a list, or the string as-is."""
    if not isinstance(paths, list):
        return paths
    for path in paths:
        check = os.path.dirname(path) if writable and os.path.dirname(path) else path
        if os.path.exists(check) and (not writable or os.access(check, os.W_OK)):
            return path
    return paths[0]


def setup_dataloaders(config):
    """Build train and validation DataLoaders from config using ChestXrayDataset."""
    data_config = config.get("data", {})
    dl_config = config.get("dataloader", {})
    transform_config = config.get("transform", {})

    train_loader, val_loader, _ = get_dataloaders(
        data_path=data_config.get("image_dir", "data/sample/images"),
        label_path=data_config.get("labels_csv", "data/sample_labels.csv"),
        batch_size=dl_config.get("batch_size", 32),
        val_split=dl_config.get("val_split", 0.2),
        test_split=dl_config.get("test_split", 0.1),
        seed=dl_config.get("seed", 42),
        num_workers=dl_config.get("num_workers", 0),
        train_transform=get_train_transform(transform_config),
        eval_transform=get_transforms(transform_config),
    )
    return train_loader, val_loader


def setup_optimizer(model, train_config):
    """Create optimizer from config."""
    optimizer_type = train_config.get("optimizer", "Adam")
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0)

    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                         momentum=train_config.get("momentum", 0.9), nesterov=True)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def setup_scheduler(optimizer, train_config):
    """Create LR scheduler from config, or return None if not configured."""
    sched_config = train_config.get("scheduler")
    if not sched_config:
        return None
    name = sched_config.get("name", "ReduceLROnPlateau")
    if name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_config.get("factor", 0.1),
            patience=sched_config.get("patience", 1),
            min_lr=sched_config.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def setup_criterion(config: dict, train_loader=None) -> nn.Module:
    """Instantiate loss function from config. Falls back to AsymmetricLoss if not specified."""
    loss_config = config.get("loss", {"name": "asymmetric"})
    return build_loss(loss_config, train_loader=train_loader)


def main():
    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument("model", type=str, help="Model name to train")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights for fine-tuning")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    model = create_model(args.model, config)

    fine_tune_cfg = config.get("fine_tune", {})
    if fine_tune_cfg.get("pretrained", False):
        model.load_imagenet_weights()
    if fine_tune_cfg.get("freeze_backbone", False):
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone frozen — {trainable:,} trainable parameters (classifier head only)")

    trainloader, valloader = setup_dataloaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    train_config = config.get("training", {})
    out_path_cfg = train_config.get("output_paths") or train_config.get("output_path", "models/checkpoints")
    output_path = resolve_path(out_path_cfg, writable=True)

    # Init W&B if available
    use_wandb = False
    if wandb is not None:
        try:
            wandb.init(project="chestxray", name=args.model, config=config)
            use_wandb = True
            print("W&B logging enabled.")
        except Exception as e:
            print(f"W&B not available, skipping: {e}")

    # Retrieve label names from dataloader dataset if available
    label_names = getattr(trainloader.dataset, "labels", None)

    optimizer = setup_optimizer(model, train_config)
    train(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        criterion=setup_criterion(config, train_loader=trainloader),
        optimizer=optimizer,
        scheduler=setup_scheduler(optimizer, train_config),
        device=device,
        output_path=output_path,
        weights_name=args.model,
        epochs=train_config.get("epochs", 20),
        patience=train_config.get("patience", 5),
        start_weights=args.weights,
        use_wandb=use_wandb,
        label_names=label_names,
    )

    print("Tuning per-label thresholds on validation set...")
    best_thresholds = tune_thresholds(model, valloader, device)
    threshold_path = os.path.join(output_path, f"{args.model}_thresholds.npy")
    np.save(threshold_path, best_thresholds)
    label_names_list = getattr(trainloader.dataset, "labels", [f"label_{i}" for i in range(len(best_thresholds))])
    for name, t in zip(label_names_list, best_thresholds):
        print(f"  {name}: {t:.2f}")
    print(f"Thresholds saved to {threshold_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()