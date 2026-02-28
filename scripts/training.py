import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


# Add the project root directory to the Python module search path to enable imports from sibling directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def get_registered_models(config=None):
    """
    Get list of model names from registry in config file.

    Args:
        config (dict, optional): Configuration dictionary that may contain a model registry
                                 under config["model_registry"] as a dict
                                 mapping model names to (module_path, class_name).

    Returns:
        list: Registered model names.
    """
    if config is not None and "model_registry" in config:
        return list(config["model_registry"].keys())

def load_config(model_name):
    """Load base config, optionally merged with model-specific config."""
    base_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Config file not found: {base_path}")
    with open(base_path, "r") as f:
        config = yaml.safe_load(f) or {}

    model_path = os.path.join("configs", f"{model_name.lower()}.yaml")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model_config = yaml.safe_load(f) or {}
        config.update(model_config)

    return config


def create_model(model_name, config):
    """Create model instance based on name and config using registry."""
    registry = config.get("model_registry", {})
    model_key = model_name.lower()

    if model_key not in registry:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(registry)}")

    entry = registry[model_key]
    module_path, class_name = entry["module_path"], entry["class_name"]

    # Dynamically import the model class
    module = __import__(module_path, fromlist=[class_name])
    ModelClass = getattr(module, class_name)

    # Instantiate model with config parameters
    model = ModelClass(**config["model_params"])

    return model


def get_transforms(config):
    """Create transforms based on config."""
    transform_list = []

    # Resize if specified
    if "img_size" in config:
        transform_list.append(transforms.Resize((config["img_size"], config["img_size"])))

    transform_list.append(transforms.ToTensor())

    # Normalize if specified
    if "normalize" in config and config["normalize"]:
        transform_list.append(
            transforms.Normalize(
                mean=config.get("mean", [0.485, 0.456, 0.406]),
                std=config.get("std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(transform_list)


    
def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sample, label in val_loader:
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total if total > 0 else 0

    return avg_loss, accuracy


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    val_loader=None,
    patience=5,
    output_path="weights",
    weights_name="final_model",
    start_weights=None,
):
    best_loss = float("inf")
    patience_counter = 0

    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device))

    # Create the full output path directory structure
    os.makedirs(output_path, exist_ok=True)
    print(f"Training model in {output_path}")

    # Create a timestamped log file for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_path, f"training_logs_{timestamp}.txt")

    # Write training start info
    with open(log_path, "w") as the_file:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training started at: {timestamp_str}\n")
        the_file.write(f"Epochs: {epochs}, Patience: {patience}\n")
        the_file.write(f"Output path: {output_path}\n")
        the_file.write(f"Device: {device}\n")
        the_file.write("-" * 50 + "\n")

    print('start training')

    for epoch in range(epochs):
        checkpoint1 = time.time()
        epoch_loss = 0
        num_batches = 0
        model.train()
        for sample, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move data to device
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)
        checkpoint2 = time.time()
        epoch_time = checkpoint2 - checkpoint1

        # Run validation if val_loader is provided
        if val_loader is not None:
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%")

            # Save training logs with validation metrics
            with open(log_path, "a") as the_file:
                the_file.write(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.2f}%\n")
                time_mins = epoch_time / 60
                the_file.write(f"Epoch {epoch+1}/{epochs}, needed {time_mins:.2f} minutes\n")

            # Early stopping based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model in the same directory as final model
                best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with val_loss={val_loss:.4f}: {best_model_path}")
            else:
                patience_counter += 1
        else:
            # No validation set - use training loss for early stopping
            print(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}")

            # Save training logs without validation metrics
            with open(log_path, "a") as the_file:
                the_file.write(f"Epoch {epoch+1}/{epochs}, train_loss={avg_train_loss:.4f}\n")
                time_mins = epoch_time / 60
                the_file.write(f"Epoch {epoch+1}/{epochs}, needed {time_mins:.2f} minutes\n")

            # Early stopping based on training loss
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
                # Save best model in the same directory as final model
                best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with train_loss={avg_train_loss:.4f}: {best_model_path}")
            else:
                patience_counter += 1

        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model in the same directory as best models
    final_model_path = os.path.join(output_path, f"{weights_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Write training completion info to log
    with open(log_path, "a") as the_file:
        the_file.write("-" * 50 + "\n")
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training completed at: {completion_time}\n")
        the_file.write(f"Best loss achieved: {best_loss:.4f}\n")
        the_file.write(f"Final model saved: {final_model_path}\n")



def resolve_path(paths, default, writable=False):
    """Return the first valid path from a list, or the string as-is."""
    if not isinstance(paths, list):
        return paths
    for path in paths:
        check = os.path.dirname(path) if writable and os.path.dirname(path) else path
        if os.path.exists(check) and (not writable or os.access(check, os.W_OK)):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))] if not writable and os.path.isdir(path) else [True]
            if subdirs:
                print(f"Using path: {path}")
                return path
    print(f"Warning: no valid path found, using: {paths[0]}")
    return paths[0]


def setup_dataloaders(config, transform):
    """Build train and optional validation DataLoaders from config."""
    data_config = config.get("data", {})
    root_path = resolve_path(data_config.get("root", "data/Dataset"), "data/Dataset")
    full_dataset = datasets.ImageFolder(root=root_path, transform=transform)

    batch_size = data_config.get("batch_size", 32)
    val_split = data_config.get("val_split", 0.2)

    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        trainset, valset = random_split(full_dataset, [train_size, val_size], generator=generator)
        print(f"Dataset split: {train_size} training, {val_size} validation samples")
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    else:
        trainset = full_dataset
        valloader = None
        print(f"No validation split. Using all {len(trainset)} samples for training.")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=data_config.get("shuffle", True))
    return trainloader, valloader


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


def main():
    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument("model", type=str, help="Model name to train")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights for fine-tuning")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    model = create_model(args.model, config)
    trainloader, valloader = setup_dataloaders(config, get_transforms(config.get("transforms", {})))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    train_config = config.get("training", {})
    output_path = resolve_path(
        train_config.get("output_paths", train_config.get("output_path", "models/checkpoints")),
        "models/checkpoints", writable=True
    )

    train(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=setup_optimizer(model, train_config),
        device=device,
        output_path=output_path,
        weights_name=args.model,
        epochs=train_config.get("epochs", 20),
        patience=train_config.get("patience", 5),
        start_weights=args.weights,
    )


if __name__ == "__main__":
    main()