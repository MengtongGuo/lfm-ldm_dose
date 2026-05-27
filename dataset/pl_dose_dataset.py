import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import pytorch_lightning as pl
#from torch.utils.data import random_split, DataLoader
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn.functional as F


class ConditionalDoseDataset(Dataset):
    def __init__(self, root_dir, target_dim=None, downsample_factor=None, normalize=True):
        """
        Args:
            root_dir: Path to the dataset.
            normalize: Whether to normalize each volume individually.
            target_dim: Target spatial dimension as single integer (e.g., 128 → (128, 128, 128))
        """
        super().__init__()
        self.data = []
        self.normalize = normalize
        self.target_dim = target_dim  # Now specifies the exact shape
        self.downsample_factor = downsample_factor


        beam_folders = sorted(os.listdir(root_dir))
        for beam_folder in tqdm(beam_folders, desc="Indexing dataset"):
            beam_path = os.path.join(root_dir, beam_folder, "output")
            if not os.path.isdir(beam_path):
                continue

            energy = float(beam_folder.replace("_", "."))

            for batch_id in os.listdir(beam_path):
                batch_path = os.path.join(beam_path, batch_id)
                input_path = os.path.join(batch_path, "input_cubes")
                output_path = os.path.join(batch_path, "output_cubes")

                input_files = sorted(glob.glob(os.path.join(input_path, "*.npy")))
                for file in input_files:
                    filename = os.path.basename(file)
                    input_file = os.path.join(input_path, filename)
                    output_file = os.path.join(output_path, filename)
                    if os.path.exists(output_file):
                        #self.data.append((input_file, output_file, energy))
                        self.data.append((input_file, output_file, energy, f"{beam_folder}_{batch_id}"))

    def __len__(self):
        return len(self.data)
    
    def reshape_tensor(self, tensor):
        """Resize tensor to target_dim or by downsample_factor."""
        d, h, w = tensor.shape[-3:]

        if getattr(self, "target_dim", None):
            size = (self.target_dim,) * 3 if isinstance(self.target_dim, int) else self.target_dim
        elif getattr(self, "downsample_factor", None):
            f = self.downsample_factor
            size = (max(1, d // f), max(1, h // f), max(1, w // f))
        else:
            return tensor

        return F.interpolate(tensor.unsqueeze(0), size=size, mode="trilinear", align_corners=False).squeeze(0)

    #def reshape_tensor(self, tensor):
        #"""Resizes a 4D tensor (1, D, H, W) to (1, target_dim, target_dim, target_dim)."""

        #if self.target_dim is None:
            #return tensor
        #return F.interpolate(tensor.unsqueeze(0), size=(self.target_dim,) * 3, mode='trilinear', align_corners=False).squeeze(0)

    def __getitem__(self, idx):
        #input_file, output_file, energy = self.data[idx]
        input_file, output_file, energy, _ = self.data[idx]
        input_vol = np.load(input_file).astype(np.float32)
        output_vol = np.load(output_file).astype(np.float32)

        if self.normalize:
            input_vol = (input_vol - input_vol.mean()) / (input_vol.std() + 1e-5)
            output_vol = (output_vol - output_vol.mean()) / (output_vol.std() + 1e-5)


        input_tensor = torch.from_numpy(input_vol).unsqueeze(0)   # Shape: [1, D, H, W]
        dose_tensor = torch.from_numpy(output_vol).unsqueeze(0)   # Shape: [1, D, H, W]

        input_tensor = self.reshape_tensor(input_tensor)
        dose_tensor = self.reshape_tensor(dose_tensor)

        condition = {
            "ct": input_tensor.clone(),
            "energy": torch.tensor([energy], dtype=torch.float32)
        }

        return dose_tensor, condition


class DoseDataModule(pl.LightningDataModule):
    def __init__(self,root_dir,batch_size=4,num_workers=4,test_ratio=0.1,num_folds=5,current_fold=0,target_dim=None,downsample_factor=None,normalize=True,dataset_fraction=0.1,seed=42,):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_ratio = test_ratio
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.target_dim = target_dim
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.dataset_fraction = dataset_fraction
        self.seed = seed

    def setup(self, stage=None):
        full_dataset = ConditionalDoseDataset(self.root_dir, target_dim=self.target_dim, downsample_factor=self.downsample_factor, normalize=self.normalize,)
        total_full = len(full_dataset)
        
        if self.dataset_fraction < 1.0:
            subset_len = int(total_full * self.dataset_fraction)
            indices = torch.randperm(total_full)[:subset_len].tolist()
            dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            dataset = full_dataset
        
        # Fix test + perform 5-fold operations on the rest
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
        else:
            base_dataset = dataset
            subset_indices = list(range(len(dataset)))

        rng = np.random.RandomState(self.seed)
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)
        
        n_test = int(len(dataset) * self.test_ratio)
        test_indices = all_indices[:n_test]
        dev_indices = all_indices[n_test:]
        
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        splits = list(kf.split(dev_indices))
        train_idx, val_idx = splits[self.current_fold]
        
        train_indices = [dev_indices[i] for i in train_idx]
        val_indices = [dev_indices[i] for i in val_idx]
        
        self.train_set = Subset(dataset, train_indices)
        self.val_set = Subset(dataset, val_indices)
        self.test_set = Subset(dataset, test_indices)

        total = len(dataset)

        # TEMP - just take 200 images for test dataset (200 images will be generated)
        #self.test_set = torch.utils.data.Subset(self.test_set, list(range(200)))

        print(f"[DEBUG] Full dataset samples: {total_full}")
        print(f"[DEBUG] Dataset fraction: {self.dataset_fraction}")
        print(f"[DEBUG] Total samples: {total}")
        print(f"[DEBUG] Train samples: {len(self.train_set)}")
        print(f"[DEBUG] Val samples: {len(self.val_set)}")
        print(f"[DEBUG] Test samples: {len(self.test_set)}")


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    # each energy has 10 batches
    # each batch has 10 patients (100 total patients)
    # 200 patients cubes per patient
    # = 160 000 sample cubes (100x100x100) in the entire dataset
    import random

    # Point this to your dataset root
    root_dir = "/hdd/Josch_Data/simulations"

    # Instantiate the dataset
    dataset = ConditionalDoseDataset(root_dir=root_dir, normalize=True)

    print(f"✅ Loaded dataset with {len(dataset)} samples")

    # Pick a random sample
    index = random.randint(0, len(dataset) - 1)
    input_tensor, condition, target_tensor = dataset[index]

    print("\n--- Sample Inspection ---")
    print(f"Index: {index}")
    print(f"Input CT shape:        {input_tensor.shape} (should be [1, D, H, W])")
    print(f"Target Dose shape:     {target_tensor.shape} (should be [1, D, H, W])")
    print(f"Condition 'ct' shape:  {condition['ct'].shape} (copy of input)")
    print(f"Condition 'energy':    {condition['energy'].item()} keV")

    assert input_tensor.shape == target_tensor.shape, "❌ Input and target shapes do not match!"
    assert condition["ct"].shape == input_tensor.shape, "❌ Condition CT shape mismatch!"
    assert isinstance(condition["energy"], torch.Tensor), "❌ Energy is not a tensor!"

    print("✅ All checks passed.")