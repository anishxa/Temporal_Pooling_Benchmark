import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import torchaudio
import pandas as pd
from scipy.io.wavfile import read as wav_read
from transformers import WavLMModel, Wav2Vec2Model, HubertModel, Data2VecAudioModel
from torch.nn.functional import cosine_similarity

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMPLES = 64  # Reduced slightly to fit all 6 models in memory/time

print(f"Device: {DEVICE}")

# Load a small batch of EDAIC val
df = pd.read_csv("data/utterance_table_edaic_segmented_split.csv")
df_val = df[df["split"] == "val"].reset_index(drop=True).head(N_SAMPLES)

def load_waveform(path):
    if path.startswith("edaic_segments/") or path.startswith("modma_segments/"):
        path = os.path.join("../WavLM_Depression_Detection", path)
    try:
        sr, data = wav_read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data)
    except:
        waveform = torch.zeros(48000)
    if waveform.shape[0] < 48000:
        waveform = torch.cat([waveform, torch.zeros(48000 - waveform.shape[0])])
    elif waveform.shape[0] > 48000:
        waveform = waveform[:48000]
    return waveform

print(f"Loading {N_SAMPLES} EDAIC val waveforms...")
waveforms = torch.stack([load_waveform(p) for p in df_val["file_path"].tolist()])
waveforms = waveforms.to(DEVICE)
# Normalize
mean = waveforms.mean(dim=-1, keepdim=True)
var = waveforms.var(dim=-1, keepdim=True)
waveforms_norm = (waveforms - mean) / torch.sqrt(var + 1e-7)

models_to_test = {
    "wavlm-base-plus": ("microsoft/wavlm-base-plus", WavLMModel),
    "wavlm-large": ("microsoft/wavlm-large", WavLMModel),
    "xls-r-1b": ("facebook/wav2vec2-xls-r-1b", Wav2Vec2Model),
    "hubert-large": ("facebook/hubert-large-ls960-ft", HubertModel),
    "w2v2-robust": ("facebook/wav2vec2-large-robust", Wav2Vec2Model),
    "data2vec-large": ("facebook/data2vec-audio-large-960h", Data2VecAudioModel),
}

for name, (path, model_class) in models_to_test.items():
    print(f"\n==========================================")
    print(f"Testing model: {name}")
    print(f"==========================================")
    
    print("Loading fp32 model...")
    model_fp32 = model_class.from_pretrained(path, output_hidden_states=True).to(DEVICE).eval()
    
    print("Running fp32 extraction...")
    with torch.no_grad():
        out32 = model_fp32(input_values=waveforms_norm)
        feat32 = out32.hidden_states[-1].mean(dim=1).cpu().float()
    
    del model_fp32
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("Loading fp16 model...")
    model_fp16 = model_class.from_pretrained(path, output_hidden_states=True).to(DEVICE).half().eval()
    
    print("Running fp16 extraction...")
    with torch.no_grad():
        out16 = model_fp16(input_values=waveforms_norm.half())
        feat16 = out16.hidden_states[-1].float().mean(dim=1).cpu()
    
    del model_fp16
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    cos_sims = cosine_similarity(feat32, feat16, dim=1)
    print(f"\n--- SANITY CHECK RESULTS ({name}) ---")
    print(f"Cosine similarity: mean={cos_sims.mean():.6f}  min={cos_sims.min():.6f}  max={cos_sims.max():.6f}")
    print(f"L2 distance:       mean={torch.norm(feat32 - feat16, dim=1).mean():.6f}")
    print(f"Max absolute diff: {(feat32 - feat16).abs().max():.6f}")
    
    threshold = 0.999
    if cos_sims.min() >= threshold:
        print(f"✅ PASS: min cosine sim {cos_sims.min():.6f} >= {threshold}. fp16 safe.")
    else:
        print(f"❌ FAIL: min cosine sim {cos_sims.min():.6f} < {threshold}. DO NOT USE fp16.")
