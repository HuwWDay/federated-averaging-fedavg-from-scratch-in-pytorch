"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    # TODO: return an nn.Module mapping (N, input_size) floats to (N, num_classes) logits
    mlp = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),  
        nn.Linear(hidden_size, num_classes),
    )
    return mlp

# Step 2 - build_synthetic_dataset
import torch


def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    # TODO: build a seeded synthetic dataset of (features, labels) tensors

    generator = torch.Generator()
    generator.manual_seed(seed)

    features = torch.randn(
        num_samples, input_size, generator=generator, dtype=torch.float32
    )

    labels = torch.randint(
        0, num_classes, (num_samples,), generator=generator, dtype=torch.long
    )

    return features, labels

# Step 3 - train_test_split_dataset
import torch


def train_test_split_dataset(features, labels, test_fraction, seed):
    # TODO: seeded shuffle of row indices, then slice into train and test sets

    num_samples = features.shape[0]

    num_test = int(num_samples * test_fraction)
    num_train = num_samples - num_test

    generator = torch.Generator()
    generator.manual_seed(seed)

    shuffled_indices = torch.randperm(num_samples, generator=generator)

    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]

    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
import torch


def partition_data_iid(train_features, train_labels, num_clients, seed):
    # TODO: Shuffle the M rows with seed and split them evenly across num_clients clients.

    num_samples = train_features.shape[0]

    # Create a local generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Generate a shuffled sequence of indices covering all M rows
    shuffled_indices = torch.randperm(num_samples, generator=generator)

    if num_clients == 0:
        # Return the entire shuffled dataset as a single partition
        return [
            (train_features[shuffled_indices], train_labels[shuffled_indices])
        ]
    # -------------------------------------------------------

    # Calculate base size and the remainder for clean, uneven distribution
    base_size = num_samples // num_clients
    remainder = num_samples % num_clients

    client_data = []
    current_idx = 0

    for i in range(num_clients):
        # Dynamically distribute the remainder rows across the first few clients
        client_size = base_size + (1 if i < remainder else 0)

        # Slice the appropriate block of shuffled indices
        start_idx = current_idx
        end_idx = current_idx + client_size
        indices_for_client = shuffled_indices[start_idx:end_idx]

        # Extract features and labels for this specific client
        client_feat = train_features[indices_for_client]
        client_lab = train_labels[indices_for_client]

        # Append as a tuple pair
        client_data.append((client_feat, client_lab))

        # Advance the pointer for the next client
        current_idx = end_idx

    return client_data

# Step 5 - partition_data_non_iid
import torch


def partition_data_non_iid(
    train_features, train_labels, num_clients, shards_per_client, seed
):
    # TODO: Sort the data by label and assign label-contiguous shards to each client.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Handle the zero client edge case 
    if num_clients == 0:
        
        shuffled_indices = torch.randperm(
            train_features.shape[0], generator=generator
        )
        return [
            (train_features[shuffled_indices], train_labels[shuffled_indices])
        ]

    num_samples = train_features.shape[0]
    total_shards = num_clients * shards_per_client
    shard_size = num_samples // total_shards

    # Sort the entire dataset by label to group similar classes together
    sorted_indices = torch.argsort(train_labels)
    sorted_features = train_features[sorted_indices]
    sorted_labels = train_labels[sorted_indices]

    # Use the seeded generator to shuffle the shard ordering deterministically
    shard_order = torch.randperm(total_shards, generator=generator)

    client_data = []

    # Assign shards to each client sequentially based on the shuffled order
    for client_idx in range(num_clients):
        client_feats_list = []
        client_labs_list = []

        # Gather the specific shards assigned to this client
        for s in range(shards_per_client):
            shard_idx = shard_order[client_idx * shards_per_client + s]

            start_sample = shard_idx * shard_size
            # The last shard absorbs any leftover rounding remainder samples
            if shard_idx == total_shards - 1:
                end_sample = num_samples
            else:
                end_sample = start_sample + shard_size

            client_feats_list.append(sorted_features[start_sample:end_sample])
            client_labs_list.append(sorted_labels[start_sample:end_sample])

        # Concatenate this client's shards back into unified tensors
        client_features = torch.cat(client_feats_list, dim=0)
        client_labels = torch.cat(client_labs_list, dim=0)

        client_data.append((client_features, client_labels))

    return client_data

# Step 6 - count_client_samples (not yet solved)
# TODO: implement

# Step 7 - iterate_client_batches (not yet solved)
# TODO: implement

# Step 8 - compute_batch_loss (not yet solved)
# TODO: implement

# Step 9 - local_sgd_step (not yet solved)
# TODO: implement

# Step 10 - train_client_local (not yet solved)
# TODO: implement

# Step 11 - clone_model_state (not yet solved)
# TODO: implement

# Step 12 - load_model_state (not yet solved)
# TODO: implement

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

