import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
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
    """Create transforms based on config."""
    transform_list = []

    if "image_size" in config:
        transform_list.append(transforms.Resize((config["image_size"], config["image_size"])))

    transform_list.append(transforms.ToTensor())

    if "normalize" in config and config["normalize"]:
        transform_list.append(
            transforms.Normalize(
                mean=config.get("mean", [0.485, 0.456, 0.406]),
                std=config.get("std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(transform_list)


def validate(model, val_loader, criterion, device, threshold=0.5):
    """
    Validate the model on validation set.

    Returns:
        tuple: (avg_loss, exact_match_accuracy, macro_f1)
            - exact_match_accuracy: % of samples where ALL labels are correct
            - macro_f1: average F1 across all labels
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample, label in val_loader:
            sample = sample.to(device)
            label = label.to(device).float()

            logits = model(sample)
            val_loss += criterion(logits, label).item()

            preds = (torch.sigmoid(logits) >= threshold).float()
            all_preds.append(preds.cpu())
            all_targets.append(label.cpu())

    avg_loss = val_loss / len(val_loader)

    preds_cat = torch.cat(all_preds)     # (N, C)
    targets_cat = torch.cat(all_targets) # (N, C)

    # Exact match accuracy (all labels must match)
    exact_match = (preds_cat == targets_cat).all(dim=1).float().mean().item() * 100

    # Macro F1 across labels
    tp = (preds_cat * targets_cat).sum(dim=0)
    fp = (preds_cat * (1 - targets_cat)).sum(dim=0)
    fn = ((1 - preds_cat) * targets_cat).sum(dim=0)
    macro_f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item() * 100

    return avg_loss, exact_match, macro_f1


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
):
    best_loss = float("inf")
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

        if val_loader is not None:
            val_loss, exact_match, macro_f1 = validate(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            monitor_loss = val_loss
            log_line = (f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f},"
                        f" val_loss={val_loss:.4f}, exact_match={exact_match:.2f}%, macro_f1={macro_f1:.2f}%")
        else:
            monitor_loss = avg_train_loss
            log_line = f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}"

        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
            f.write(f"Epoch {epoch+1}/{epochs}, needed {epoch_time / 60:.2f} minutes\n")

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (loss={monitor_loss:.4f}): {best_model_path}")
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
        f.write(f"Best loss achieved: {best_loss:.4f}\n")
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


def setup_dataloaders(config, transform):
    """Build train and validation DataLoaders from config using ChestXrayDataset."""
    data_config = config.get("data", {})
    dl_config = config.get("dataloader", {})

    train_loader, val_loader, _ = get_dataloaders(
        data_path=data_config.get("image_dir", "data/sample/images"),
        label_path=data_config.get("labels_csv", "data/sample_labels.csv"),
        batch_size=dl_config.get("batch_size", 32),
        val_split=dl_config.get("val_split", 0.2),
        test_split=dl_config.get("test_split", 0.1),
        seed=dl_config.get("seed", 42),
        num_workers=dl_config.get("num_workers", 0),
        transform=transform,
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

    trainloader, valloader = setup_dataloaders(config, get_transforms(config.get("transform", {})))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    train_config = config.get("training", {})
    out_path_cfg = train_config.get("output_paths") or train_config.get("output_path", "models/checkpoints")
    output_path = resolve_path(out_path_cfg, writable=True)

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
    )


if __name__ == "__main__":
    main()