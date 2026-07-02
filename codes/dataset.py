import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def parse_path(path):
    base = os.path.basename(path)
    parts = re.findall(r'\d+', base)
    if len(parts) >= 3:
        speaker_id = parts[0]
        utt_idx = int(parts[1])
        seg_idx = int(parts[2])
        return speaker_id, (utt_idx, seg_idx)
    elif len(parts) == 2:
        return parts[0], (int(parts[1]), 0)
    return None, None

def build_sequences(feature_dir, metadata_csv, split, max_len=150):
    df = pd.read_csv(metadata_csv)
    df_split = df[df["split"] == split].reset_index(drop=True)
    
    if len(df_split) == 0:
        return None, None, None, None
        
    X_path = os.path.join(feature_dir, f"X_{split}_all_layers.npy")
    if not os.path.exists(X_path):
        print(f"Warning: {X_path} not found.")
        return None, None, None
        
    X = np.load(X_path) # Shape: [N, num_layers, feature_dim]
    
    # Group segments
    speaker_data = {}
    for i, row in df_split.iterrows():
        path = row["file_path"]
        spk_id, sort_key = parse_path(path)
        if spk_id is None:
            continue
        label = row["label"]
        feat = X[i]
        
        if spk_id not in speaker_data:
            speaker_data[spk_id] = {
                "feats": [],
                "sort_keys": [],
                "label": label
            }
        speaker_data[spk_id]["feats"].append(feat)
        speaker_data[spk_id]["sort_keys"].append(sort_key)
        
    # Build sequences
    sequences = []
    labels = []
    speaker_ids = []
    for spk_id, data in speaker_data.items():
        # Sort indices based on chronological sort_keys
        sorted_indices = sorted(range(len(data["sort_keys"])), key=lambda k: data["sort_keys"][k])
        sorted_feats = [data["feats"][idx] for idx in sorted_indices]
        sequences.append(np.array(sorted_feats))
        labels.append(data["label"])
        
        # Clean speaker ID to extract digits for integer format
        digits = re.findall(r'\d+', str(spk_id))
        spk_num = int(digits[0]) if digits else 0
        speaker_ids.append(spk_num)
        
    num_sequences = len(sequences)
    num_layers = X.shape[1]
    feature_dim = X.shape[2]
    
    padded_sequences = np.zeros((num_sequences, max_len, num_layers, feature_dim), dtype=np.float32)
    masks = np.zeros((num_sequences, max_len), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
        masks[i, :seq_len] = 1.0
        
    return padded_sequences, np.array(labels), masks, np.array(speaker_ids, dtype=np.int32)

def get_dataloaders(dataset_name="edaic", model_name="wavlm-base-plus", max_len=150, batch_size=16):
    base_dir = "."
    feature_dir = os.path.join(base_dir, f"features/{dataset_name}/{model_name}")
    metadata_csv = os.path.join(base_dir, f"data/utterance_table_{dataset_name}_segmented_split.csv")
    
    X_train, y_train, mask_train, spk_train = build_sequences(feature_dir, metadata_csv, "train", max_len)
    X_val, y_val, mask_val, spk_val = build_sequences(feature_dir, metadata_csv, "val", max_len)
    X_test, y_test, mask_test, spk_test = build_sequences(feature_dir, metadata_csv, "test", max_len)
    
    if X_train is None or X_test is None:
        raise FileNotFoundError(f"Feature files not found in {feature_dir}. Please extract them first.")
        
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long(), torch.tensor(mask_train), torch.tensor(spk_train).int())
    
    if X_val is not None:
        val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).long(), torch.tensor(mask_val), torch.tensor(spk_val).int())
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        # Fallback to test loader if val is not available
        val_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long(), torch.tensor(mask_test), torch.tensor(spk_test).int())
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long(), torch.tensor(mask_test), torch.tensor(spk_test).int())
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_train
