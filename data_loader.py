"""
Data loading utilities for ZKP Federated Evaluation.
Handles MNIST and UCI HAR dataset loading and client data distribution.
"""

import os
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_DIR = './data'

def download_and_extract_har(url, dest_path):
    """Downloads and extracts the HAR dataset."""
    if os.path.exists(os.path.join(dest_path, 'UCI HAR Dataset')):
        print("HAR dataset already downloaded and extracted.")
        return os.path.join(dest_path, 'UCI HAR Dataset')

    print(f"Downloading HAR dataset from {url}...")
    response = requests.get(url, stream=True, timeout=(10,120))
    response.raise_for_status() # Raise an exception for bad status codes

    try:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(dest_path)
        print(f"HAR dataset extracted to {dest_path}")
        return os.path.join(dest_path, 'UCI HAR Dataset')
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file or was corrupted.")
        return None
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return None

def load_mnist(data_dir='./data'):
    """
    Load MNIST dataset and apply transformations.
    
    Args:
        data_dir: Directory to store/load MNIST data
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

class HARDataset(Dataset):
    """Dataset class for UCI Human Activity Recognition dataset."""
    
    def __init__(self, data_dir, train=True):
        """
        Initialize HAR dataset from UCI repository files.
        
        Args:
            data_dir: Root directory containing UCI HAR Dataset
            train: If True, load training data; otherwise load test data
        """
        subset = 'train' if train else 'test'
        data_path = os.path.join(data_dir, 'UCI HAR Dataset', subset, f'X_{subset}.txt')
        labels_path = os.path.join(data_dir, 'UCI HAR Dataset', subset, f'y_{subset}.txt')
        
        self.features = pd.read_csv(data_path, sep=r'\s+', header=None).values
        self.labels = pd.read_csv(labels_path, header=None).values.reshape(-1) - 1
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        return self.features[idx], self.labels[idx]

def load_har(data_dir='./data'):
    """
    Load UCI HAR dataset.
    
    Args:
        data_dir: Directory containing UCI HAR Dataset
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_dataset = HARDataset(data_dir, train=True)
    test_dataset = HARDataset(data_dir, train=False)
    return train_dataset, test_dataset

def create_client_dataloaders(dataset, num_clients, batch_size, data_percentage=1.0):
    """
    Split dataset into client subsets and create DataLoader for each.
    
    Args:
        dataset: PyTorch Dataset to split
        num_clients: Number of clients to distribute data to
        batch_size: Batch size for DataLoaders
        data_percentage: Fraction of total data to use (0.0-1.0)
        
    Returns:
        list: List of DataLoader objects, one per client
    """
    total_size = len(dataset)
    use_size = int(total_size * data_percentage)
    if use_size < num_clients:
        raise ValueError(f"Not enough samples ({use_size}) to distribute among {num_clients} clients")

    # Sample a subset of indices from the *original* dataset
    selected_indices = torch.randperm(total_size)[:use_size].tolist()

    # Split indices among clients (distribute remainder to first clients)
    samples_per_client = use_size // num_clients
    remainder = use_size % num_clients
    client_dataloaders = []
    start_idx = 0

    for i in range(num_clients):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + samples_per_client + extra
        client_indices = selected_indices[start_idx:end_idx]
        start_idx = end_idx

        client_dataset = Subset(dataset, client_indices)
        client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(client_dataloader)

    return client_dataloaders

def get_data_loaders(dataset_name, num_clients, batch_size, data_percentage=1.0, non_iid=False, server_train_fraction=0.1):
    """
    Gets data loaders for server training and client evaluation.
    Returns: (server_train_loader, client_eval_loaders)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    client_eval_loaders = []
    server_train_loader = None

    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset = load_mnist(DATA_DIR)
        
        # Server training data (fraction of MNIST train set)
        server_train_size = int(len(train_dataset) * server_train_fraction)
        server_train_indices = np.random.choice(len(train_dataset), server_train_size, replace=False)
        server_train_subset = Subset(train_dataset, server_train_indices)
        server_train_loader = DataLoader(server_train_subset, batch_size=batch_size, shuffle=True)
        print(f"Server training using {server_train_size} MNIST samples.")

        # Client evaluation data (fraction of MNIST test set)
        client_total_eval_size = int(len(test_dataset) * data_percentage)
        client_eval_indices_all = np.random.choice(len(test_dataset), client_total_eval_size, replace=False)
        client_eval_dataset = Subset(test_dataset, client_eval_indices_all)
        print(f"Clients evaluating using {client_total_eval_size} MNIST test samples in total.")

    elif dataset_name.lower() == 'har':
        har_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        har_path = download_and_extract_har(har_url, DATA_DIR)
        if not har_path:
             raise RuntimeError("Failed to download or extract HAR dataset.")
        train_dataset, test_dataset = load_har(DATA_DIR)

        # Server training data (fraction of HAR train set)
        server_train_size = int(len(train_dataset) * server_train_fraction)
        server_train_indices = np.random.choice(len(train_dataset), server_train_size, replace=False)
        server_train_subset = Subset(train_dataset, server_train_indices)
        server_train_loader = DataLoader(server_train_subset, batch_size=batch_size, shuffle=True)
        print(f"Server training using {server_train_size} HAR train samples.")

        # Client evaluation data (fraction of HAR test set)
        client_total_eval_size = int(len(test_dataset) * data_percentage)
        client_eval_indices_all = np.random.choice(len(test_dataset), client_total_eval_size, replace=False)
        client_eval_dataset = Subset(test_dataset, client_eval_indices_all)
        print(f"Clients evaluating using {client_total_eval_size} HAR test samples in total.")

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Partition client evaluation data among clients
    all_indices = list(range(len(client_eval_dataset)))
    np.random.shuffle(all_indices)  # Shuffle for random distribution

    if non_iid:
        # Simple Non-IID simulation (e.g., by sorting data by label)
        print("Warning: Using a very basic non-IID split based on sorting.")
        if dataset_name.lower() == 'mnist':
            # Labels in the *subset order* (0..len(client_eval_dataset)-1)
            labels = test_dataset.targets[client_eval_indices_all].cpu().numpy()
        elif dataset_name.lower() == 'har':
            # HARDataset stores labels as a torch tensor
            labels = test_dataset.labels[client_eval_indices_all].cpu().numpy()
        else:
            labels = np.array([label for _, label in client_eval_dataset])

        # Sort the shuffled indices by their corresponding label in subset order
        all_indices.sort(key=lambda subset_idx: labels[subset_idx])

    num_items_per_client = len(client_eval_dataset) // num_clients
    remainder = len(client_eval_dataset) % num_clients

    idx_offset = 0
    for i in range(num_clients):
        extra = 1 if i < remainder else 0
        end_idx = idx_offset + num_items_per_client + extra
        client_indices_subset = all_indices[idx_offset:end_idx]
        idx_offset = end_idx

        # Create subset using indices relative to the client_eval_dataset
        client_subset = Subset(client_eval_dataset, client_indices_subset)

        # Ensure batch_size isn't larger than the dataset size
        actual_batch_size = min(batch_size, len(client_subset))
        if actual_batch_size == 0:
             print(f"Warning: Client {i} has no data for evaluation batch.")
             client_eval_loaders.append(None) # Or handle appropriately
             continue

        client_loader = DataLoader(client_subset, batch_size=actual_batch_size, shuffle=False)
        client_eval_loaders.append(client_loader)
        # Report the number of samples assigned to this client for evaluation
        print(f"Client {i}: {len(client_indices_subset)} evaluation samples.")

    return server_train_loader, client_eval_loaders
