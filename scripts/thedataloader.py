
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


from PIL import Image
import pandas as pd
import itertools
from collections import defaultdict
import os

#need to test id leakage and class balance across splits, then compute weights for loss

class ChestXrayDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-ray multi-label classification."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(set(itertools.chain.from_iterable(self.metadata["Finding Labels"].str.split("|"))))
        for disease in self.labels:
            self.metadata[disease] = self.metadata["Finding Labels"].str.split("|").apply(lambda x: int(disease in x))
        self.patient_ids = self.metadata["Patient ID"]
        print(f" {len(self.labels)} desease classes: {self.labels} ")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Image Index"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        label = torch.tensor([1 if value in row["Finding Labels"].split("|") else 0 for value in  self.labels], dtype=torch.bool)
        return image, label


def split_data(full_dataset, seed, val_split, test_split):
    """
    Patient-level splitting (prevents data leakage) using train_test_split.
    """
    patient_ids = full_dataset.patient_ids.unique()

    trainval_ids, test_ids = train_test_split(
        patient_ids, 
        test_size=test_split, 
        random_state=seed,
        shuffle=True
    )

    val_size_relative = val_split / (1.0 - test_split)  # how much of trainval ends up as val
    train_ids, val_ids = train_test_split(
        trainval_ids, 
        test_size=val_size_relative,
        random_state=seed,
        shuffle=True
    )

    # Map from patient_id to all their sample (row) indices
    patient_to_indices = defaultdict(list)
    for idx, pat_id in enumerate(full_dataset.patient_ids):
        patient_to_indices[pat_id].append(idx)

    # Collect sample indices for each split
    train_indices = list(itertools.chain.from_iterable(patient_to_indices[patient_id] for patient_id in train_ids))
    val_indices = list(itertools.chain.from_iterable(patient_to_indices[patient_id] for patient_id in val_ids))
    test_indices = list(itertools.chain.from_iterable(patient_to_indices[patient_id] for patient_id in test_ids))

    return train_indices, val_indices, test_indices


def get_dataloaders(data_path, label_path, batch_size=32, val_split=0.2, test_split=0.1, seed=42, num_workers=0):
    full_dataset = ChestXrayDataset(label_path, data_path)

    train_indices, val_indices, test_indices = split_data(full_dataset, seed, val_split, test_split)

    # Now make Subsets
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders('/Users/philippevannson/Desktop/super-couscous/data/sample/images','/Users/philippevannson/Desktop/super-couscous/data/sample_labels.csv')
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")







