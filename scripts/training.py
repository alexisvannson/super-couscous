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

# Add project root to path
# Add the project root directory to the Python module search path to enable imports from sibling directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Model Registry: Maps model names to (module_path, class_name)


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
    """Load configuration for a specific model."""
    config_path = os.path.join("config", f"{model_name.lower()}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def create_model(model_name, config):
    MODEL_REGISTRY = get_registered_models(config)
    """Create model instance based on name and config using registry."""
    model_key = model_name.lower()

    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(get_registered_models())
        raise ValueError(f"Unknown model: {model_name}. " f"Available models: {available_models}")

    # Get module and class name from registry
    module_path, class_name = MODEL_REGISTRY[model_key]

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





def main():
    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument("--model", type=str, required=True, help="Model name to train")
    parser.add_argument(
        "--config", type=str, default=None, help="Override default config file path"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to pretrained weights for fine-tuning"
    )
    args = parser.parse_args()

    model_name = args.model

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(model_name)

    # Create model
    model = create_model(model_name, config)

    # Setup transforms
    transform = get_transforms(config.get("transforms", {}))

    data_config = config.get("data", {})
    root_path = data_config.get("root", "data/Dataset")

    # Handle case where root is a list (for Colab/local compatibility)
    if isinstance(root_path, list):
        # Try to find the first path that exists and contains subdirectories (class folders)
        for path in root_path:
            if os.path.exists(path) and os.path.isdir(path):
                # Check if directory has subdirectories (required for ImageFolder)
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    root_path = path
                    print(f"Using dataset path: {path}")
                    break
        else:
            # If none exist with subdirs, use the first one
            root_path = root_path[0]
            print(f"Warning: No valid dataset path found. Using: {root_path}")

    # Load full dataset
    full_dataset = datasets.ImageFolder(
        root=root_path, transform=transform
    )

    # Split dataset into train and validation sets
    val_split = data_config.get("val_split", 0.2)  # Default to 20% validation
    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        # Use a fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        trainset, valset = random_split(full_dataset, [train_size, val_size], generator=generator)

        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

        # Create dataloaders
        trainloader = DataLoader(
            trainset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=data_config.get("shuffle", True),
        )

        valloader = DataLoader(
            valset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=False,
        )
    else:
        # No validation split - use entire dataset for training
        trainset = full_dataset
        trainloader = DataLoader(
            trainset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=data_config.get("shuffle", True),
        )
        valloader = None
        print(f"No validation split. Using all {len(trainset)} samples for training.")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Setup training components
    train_config = config.get("training", {})
    criterion = nn.CrossEntropyLoss()

    # Create optimizer based on type
    optimizer_type = train_config.get("optimizer", "Adam")
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0)
    momentum = train_config.get("momentum", 0.9)

    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=momentum, nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Handle output_path list (for Colab/local compatibility)
    output_path = train_config.get("output_paths", train_config.get("output_path", "models/checkpoints"))
    if isinstance(output_path, list):
        # Try to find first writable path (e.g., Google Drive might be mounted)
        for path in output_path:
            parent = os.path.dirname(path) if os.path.dirname(path) else "."
            if os.path.exists(parent) and os.access(parent, os.W_OK):
                output_path = path
                print(f"Using output path: {path}")
                break
        else:
            # Use first path and let os.makedirs create it
            output_path = output_path[0]
            print(f"Using output path: {output_path}")

    # Train model
    train(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_path=output_path,
        weights_name=model_name,
        epochs=train_config.get("epochs", 20),
        patience=train_config.get("patience", 5),
        start_weights=args.weights,
    )


if __name__ == "__main__":
    main()