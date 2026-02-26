
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import itertools
import os



class ChestXrayDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-ray multi-label classification."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(set(itertools.chain.from_iterable(self.metadata["Finding Labels"].str.split("|"))))
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

def get_dataloaders(batch_size=None, val_split=None, test_split=None, seed=None, num_workers=None):
    full_dataset = ChestXrayDataset('/Users/philippevannson/Desktop/super-couscous/data/sample_labels.csv', '/Users/philippevannson/Desktop/super-couscous/data/sample/images')

    total = len(full_dataset)
    val_size = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, val_split=0.2, test_split=0.1, seed=42, num_workers=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")








