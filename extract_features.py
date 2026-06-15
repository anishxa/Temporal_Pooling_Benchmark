import os
import re
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import WavLMModel
from tqdm import tqdm

device = torch.device("cpu")
print(f"Using device: {device} for feature extraction")

print("Loading microsoft/wavlm-base-plus model...")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device).eval()

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.file_paths = df["file_path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Handle relative paths from the CSV so it can still find the raw audio correctly
        path = self.file_paths[idx]
        if path.startswith("code/") or path.startswith("data/") or path.startswith("raw_audio/"):
            path = os.path.join("../WavLM_Depression_Detection", path)
            
        label = self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze(0)
            
            if waveform.shape[0] < 48000:
                pad_len = 48000 - waveform.shape[0]
                waveform = torch.cat([waveform, torch.zeros(pad_len)], dim=0)
            elif waveform.shape[0] > 48000:
                waveform = waveform[:48000]
                
            return waveform, label, path
        except Exception as e:
            return torch.zeros(48000), label, path

def extract_for_dataset(metadata_csv, batch_size=32):
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found: {metadata_csv}")
        return
        
    df = pd.read_csv(metadata_csv)
    print(f"\n==========================================")
    print(f"Processing Independent Extraction ({len(df)} rows)")
    
    layers = [6, 7, 8, 9]
    for layer in layers:
        os.makedirs(f"features/features_edaic_layer{layer}", exist_ok=True)
        
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if len(df_split) == 0:
            continue
            
        print(f"Extracting {split} split ({len(df_split)} items)...")
        dataset = SpeechDataset(df_split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        X_accum_mean = {layer: [] for layer in layers}
        y_accum = []
        
        with torch.no_grad():
            for waveforms, labels, paths in tqdm(dataloader, desc=f"EDAIC - {split}"):
                waveforms = waveforms.to(device)
                
                mean = waveforms.mean(dim=-1, keepdim=True)
                var = waveforms.var(dim=-1, keepdim=True)
                waveforms = (waveforms - mean) / torch.sqrt(var + 1e-7)
                
                out = model(input_values=waveforms)
                
                for layer in layers:
                    layer_feat = out.hidden_states[layer]
                    mean_pooled = layer_feat.mean(dim=1).cpu().numpy()
                    X_accum_mean[layer].append(mean_pooled)
                    
                y_accum.append(labels.numpy())
                
        y_stacked = np.concatenate(y_accum, axis=0)
        for layer in layers:
            X_stacked_mean = np.concatenate(X_accum_mean[layer], axis=0)
            y_path = f"features/features_edaic_layer{layer}/y_{split}.npy"
            
            np.save(f"features/features_edaic_layer{layer}/X_{split}_mean.npy", X_stacked_mean)
            np.save(y_path, y_stacked)
            
            print(f"  Layer {layer} saved: Mean shape {X_stacked_mean.shape}")

if __name__ == "__main__":
    extract_for_dataset("utterance_table_edaic_segmented_split.csv")
    print("\nFeature extraction completed for Temporal Pooling Benchmark!")
