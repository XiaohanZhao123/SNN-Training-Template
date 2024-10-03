import json
import pickle
import warnings
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The given buffer is not writable")


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.transform = transform

        # Load metadata
        metadata = json.loads(self.txn.get("__metadata__".encode()).decode())
        self.length = metadata["dataset_length"]
        self.x_shape = tuple(metadata["x_shape"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        key = f"{index}".encode()
        value = self.txn.get(key)
        img_data, label = pickle.loads(value)

        # Convert image data directly to PyTorch tensor
        img_tensor = torch.frombuffer(img_data, dtype=torch.float32).reshape(
            self.x_shape
        )

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

    def __del__(self):
        self.env.close()


def dataset_to_lmdb(dataset: Dataset, output_path: str, write_frequency: int = 5000):
    """
    Convert any PyTorch Dataset to LMDB format with metadata, storing PyTorch tensors directly.

    Args:
        dataset (torch.utils.data.Dataset): The input PyTorch dataset.
        output_path (str): Path to save the LMDB database.
        write_frequency (int): Number of samples to process before committing to disk.
    """

    # Prepare metadata
    sample = dataset[0]
    if isinstance(sample, tuple):
        sample_data = sample[0]
        metadata = {
            "x_shape": list(sample_data.shape),
            "total_classes": (
                len(set(dataset.targets)) if hasattr(dataset, "targets") else None
            ),
            "dataset_length": len(dataset),
        }
    else:
        sample_data = sample
        metadata = {
            "x_shape": list(sample_data.shape),
            "total_classes": None,
            "dataset_length": len(dataset),
        }

    # Estimate the size of the database
    approx_size = (
        len(dataset) * (torch.prod(torch.tensor(metadata["x_shape"])).item() * 4 + 8)
        + 1000
    )  # 4 bytes per float32, 8 bytes for label, 1000 bytes for metadata

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create LMDB environment
    env = lmdb.open(output_path, map_size=int(approx_size * 1.3))  # Use 2x for safety

    # Start the initial transaction
    txn = env.begin(write=True)

    # Store metadata
    txn.put("__metadata__".encode(), json.dumps(metadata).encode())

    try:
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]

            if isinstance(sample, tuple):
                data, label = sample
            else:
                data, label = sample, None

            # Ensure data is a PyTorch tensor
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data)
            img_data = data.cpu().numpy().tobytes()

            key = f"{idx}".encode()
            value = pickle.dumps((img_data, label))
            txn.put(key, value)

            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = env.begin(write=True)

        # Commit any remaining data
        txn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        txn.abort()
    finally:
        env.close()

    print(f"Conversion complete. LMDB database saved at {output_path}")


def verify_lmdb(lmdb_path: str, original_dataset: Dataset):
    """
    Verify the contents of the LMDB database against the original dataset.

    Args:
        lmdb_path (str): Path to the LMDB database.
        original_dataset (torch.utils.data.Dataset): The original dataset for comparison.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # Get metadata
        metadata = json.loads(txn.get("__metadata__".encode()).decode())
        print("Metadata:")
        print(json.dumps(metadata, indent=2))

        # Check all samples
        for i in tqdm(range(metadata["dataset_length"])):
            key = f"{i}".encode()
            value = txn.get(key)
            if value is None:
                print(f"Sample {i} not found in LMDB")
                continue

            img_data, lmdb_label = pickle.loads(value)

            # Convert image data to PyTorch tensor
            lmdb_tensor = torch.frombuffer(img_data, dtype=torch.float32).reshape(
                metadata["x_shape"]
            )

            # Get the original sample
            original_sample = original_dataset[i]
            if isinstance(original_sample, tuple):
                original_tensor, original_label = original_sample
            else:
                original_tensor, original_label = original_sample, None

            # Ensure original_tensor is a PyTorch tensor
            if not isinstance(original_tensor, torch.Tensor):
                original_tensor = torch.tensor(original_tensor)

            # Compare tensors
            if not torch.allclose(lmdb_tensor, original_tensor):
                print(f"Mismatch in sample {i}")
                print(f"LMDB tensor: {lmdb_tensor}")
                print(f"Original tensor: {original_tensor}")

            # Compare labels if they exist
            if (
                lmdb_label is not None
                and original_label is not None
                and lmdb_label != original_label
            ):
                print(
                    f"Label mismatch in sample {i}: LMDB {lmdb_label}, Original {original_label}"
                )

            # Print progress every 1000 samples
            if i % 1000 == 0:
                print(f"Verified {i} samples...")

    env.close()
    print("Verification complete.")
