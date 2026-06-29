import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import WavLMModel, Wav2Vec2Model, HubertModel, Data2VecAudioModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="All-Layer Feature Extraction for Temporal Pooling Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="wavlm-base-plus",
        choices=["wavlm-base-plus", "wavlm-large", "xls-r-1b", "hubert-large", "w2v2-robust", "data2vec-large"],
        help="Speech SSL model to extract features from"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="edaic",
        choices=["edaic", "modma"],
        help="Dataset name (edaic or modma)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction"
    )
    args = parser.parse_args()

    # === Device selection ===
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} for feature extraction")

    # === Model selection ===
    model_paths = {
        "wavlm-base-plus": "microsoft/wavlm-base-plus",
        "wavlm-large": "microsoft/wavlm-large",
        "xls-r-1b": "facebook/wav2vec2-xls-r-1b",
        "hubert-large": "facebook/hubert-large-ls960-ft",
        "w2v2-robust": "facebook/wav2vec2-large-robust",
        "data2vec-large": "facebook/data2vec-audio-large-960h",
    }
    model_path = model_paths[args.model]
    print(f"Loading pre-trained model: {model_path} ...")

    if "wavlm" in args.model:
        model = WavLMModel.from_pretrained(model_path, output_hidden_states=True).to(device).eval()
    elif "xls-r" in args.model or "w2v2" in args.model:
        model = Wav2Vec2Model.from_pretrained(model_path, output_hidden_states=True).to(device).eval()
    elif "hubert" in args.model:
        model = HubertModel.from_pretrained(model_path, output_hidden_states=True).to(device).eval()
    elif "data2vec" in args.model:
        model = Data2VecAudioModel.from_pretrained(model_path, output_hidden_states=True).to(device).eval()

    # === Dataset settings ===
    metadata_csv = f"utterance_table_{args.dataset}_segmented_split.csv"
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file {metadata_csv} not found in the current directory.")

    output_dir = f"features/{args.dataset}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    class SpeechDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.file_paths = df["file_path"].tolist()
            self.labels = df["label"].tolist()

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            path = self.file_paths[idx]
            # Prepend WavLM_Depression_Detection path if needed
            if path.startswith("code/") or path.startswith("data/") or path.startswith("raw_audio/") or path.startswith("edaic_segments/") or path.startswith("modma_segments/"):
                path = os.path.join("../WavLM_Depression_Detection", path)
                
            label = self.labels[idx]
            try:
                waveform, sr = torchaudio.load(path)
                if waveform.shape[0] == 2:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                waveform = waveform.squeeze(0)
                
                # Keep exact crop/pad behavior of the benchmark
                if waveform.shape[0] < 48000:
                    pad_len = 48000 - waveform.shape[0]
                    waveform = torch.cat([waveform, torch.zeros(pad_len)], dim=0)
                elif waveform.shape[0] > 48000:
                    waveform = waveform[:48000]
                    
                return waveform, label, path
            except Exception as e:
                # Fallback to zero waveform on error
                return torch.zeros(48000), label, path

    df = pd.read_csv(metadata_csv)
    print(f"\n==========================================")
    print(f"Processing dataset: {args.dataset} with model: {args.model} ({len(df)} rows)")
    
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if len(df_split) == 0:
            continue
            
        print(f"Extracting {split} split ({len(df_split)} items)...")
        dataset = SpeechDataset(df_split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        X_accum = []
        y_accum = []
        
        with torch.no_grad():
            for waveforms, labels, paths in tqdm(dataloader, desc=f"{args.dataset.upper()} - {split}"):
                waveforms = waveforms.to(device)
                
                # Input normalization
                mean = waveforms.mean(dim=-1, keepdim=True)
                var = waveforms.var(dim=-1, keepdim=True)
                waveforms = (waveforms - mean) / torch.sqrt(var + 1e-7)
                
                out = model(input_values=waveforms)
                
                # out.hidden_states is a tuple of length (num_layers + 1)
                # each layer is [B, T, D]
                # we mean-pool across the time dimension (dim=1) for all layers
                layer_features = []
                for layer_feat in out.hidden_states:
                    mean_pooled = layer_feat.mean(dim=1).cpu().numpy() # [B, D]
                    layer_features.append(mean_pooled)
                
                # Stack layers to shape [B, num_layers, D]
                stacked = np.stack(layer_features, axis=1)
                X_accum.append(stacked)
                y_accum.append(labels.numpy())
                
        X_stacked = np.concatenate(X_accum, axis=0)
        y_stacked = np.concatenate(y_accum, axis=0)
        
        X_path = os.path.join(output_dir, f"X_{split}_all_layers.npy")
        y_path = os.path.join(output_dir, f"y_{split}.npy")
        
        np.save(X_path, X_stacked)
        np.save(y_path, y_stacked)
        
        print(f"  Saved {split} features: shape {X_stacked.shape} to {X_path}")

    print(f"\nFeature extraction completed for dataset={args.dataset}, model={args.model}!")

if __name__ == "__main__":
    main()
