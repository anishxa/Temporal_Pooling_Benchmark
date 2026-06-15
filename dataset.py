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
        return None, None, None
        
    X_path = os.path.join(feature_dir, f"X_{split}_mean.npy")
    if not os.path.exists(X_path):
        print(f"Warning: {X_path} not found.")
        return None, None, None
        
    X = np.load(X_path)
    
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
    for spk_id, data in speaker_data.items():
        # Sort indices based on chronological sort_keys
        sorted_indices = sorted(range(len(data["sort_keys"])), key=lambda k: data["sort_keys"][k])
        sorted_feats = [data["feats"][idx] for idx in sorted_indices]
        sequences.append(np.array(sorted_feats))
        labels.append(data["label"])
        
    num_sequences = len(sequences)
    feature_dim = X.shape[1]
    
    padded_sequences = np.zeros((num_sequences, max_len, feature_dim), dtype=np.float32)
    masks = np.zeros((num_sequences, max_len), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
        masks[i, :seq_len] = 1.0
        
    return padded_sequences, np.array(labels), masks

def get_dataloaders(layer=7, max_len=150, batch_size=16):
    base_dir = "."
    feature_dir = os.path.join(base_dir, f"features/features_edaic_layer{layer}")
    metadata_csv = os.path.join(base_dir, "utterance_table_edaic_segmented_split.csv")
    
    X_train, y_train, mask_train = build_sequences(feature_dir, metadata_csv, "train", max_len)
    X_val, y_val, mask_val = build_sequences(feature_dir, metadata_csv, "val", max_len)
    X_test, y_test, mask_test = build_sequences(feature_dir, metadata_csv, "test", max_len)
    
    if X_train is None or X_test is None:
        raise FileNotFoundError(f"Feature files not found in {feature_dir}. Please extract them first.")
        
    # Combine Train + Val for full training
    X_train_all = np.concatenate([X_train, X_val], axis=0) if X_val is not None else X_train
    y_train_all = np.concatenate([y_train, y_val], axis=0) if y_val is not None else y_train
    mask_train_all = np.concatenate([mask_train, mask_val], axis=0) if mask_val is not None else mask_train
    
    train_ds = TensorDataset(torch.tensor(X_train_all), torch.tensor(y_train_all).long(), torch.tensor(mask_train_all))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long(), torch.tensor(mask_test))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, y_train_all
