import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from dataset import get_dataloaders
from models import (
    MeanPoolingNet, StatisticalPoolingNet, SelfAttentionPoolingNet, 
    TransformerPoolingNet, BiGRUPoolingNet, NetVLADPoolingNet
)

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seqs, labels, masks in test_loader:
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            logits = model(seqs, masks)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # Sensitivity (Recall of class 1 / MDD)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    # Specificity (Recall of class 0 / HC)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    return acc, f1, auc, sensitivity, specificity

def run_experiment(model_name, model, train_loader, test_loader, class_weights, device, epochs=50):
    print(f"--- Running Benchmark for: {model_name} ---")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_f1 = 0.0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for seqs, labels, masks in train_loader:
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        acc, f1, auc, sens, spec = evaluate(model, test_loader, device)
        if f1 >= best_f1:
            best_f1 = f1
            best_metrics = (acc, f1, auc, sens, spec)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val F1: {f1:.4f}")
            
    if best_metrics is None:
        best_metrics = (acc, f1, auc, sens, spec)
        
    print(f"Best metrics for {model_name}: Acc={best_metrics[0]:.4f}, F1={best_metrics[1]:.4f}\n")
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="Run Temporal Pooling Benchmark under model and dataset settings")
    parser.add_argument(
        "--model",
        type=str,
        default="wavlm-base-plus",
        choices=["wavlm-base-plus", "wavlm-large", "xls-r-1b", "hubert-large", "w2v2-robust", "data2vec-large"],
        help="Speech SSL model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="edaic",
        choices=["edaic", "modma"],
        help="Dataset name (edaic or modma)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs to train downstream models"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=150,
        help="Max sequence length of pooled clips"
    )
    args = parser.parse_args()

    # === Device selection ===
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training downstream models")

    print(f"Loading data for dataset={args.dataset}, model={args.model}...")
    try:
        train_loader, test_loader, y_train_all = get_dataloaders(
            dataset_name=args.dataset,
            model_name=args.model,
            max_len=args.max_len,
            batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(e)
        return
        
    class_counts = torch.bincount(torch.tensor(y_train_all).long())
    class_weights = len(y_train_all) / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    # Model dimensions
    model_dims = {
        "wavlm-base-plus": {"num_layers": 13, "input_dim": 768},
        "wavlm-large": {"num_layers": 25, "input_dim": 1024},
        "xls-r-1b": {"num_layers": 49, "input_dim": 1280},
        "hubert-large": {"num_layers": 25, "input_dim": 1024},
        "w2v2-robust": {"num_layers": 25, "input_dim": 1024},
        "data2vec-large": {"num_layers": 25, "input_dim": 1024},
    }
    
    dims = model_dims[args.model]
    num_layers = dims["num_layers"]
    input_dim = dims["input_dim"]

    models_to_test = {
        "Mean Pooling": MeanPoolingNet(num_layers, input_dim).to(device),
        "Statistical Pooling": StatisticalPoolingNet(num_layers, input_dim).to(device),
        "Self-Attention": SelfAttentionPoolingNet(num_layers, input_dim).to(device),
        "Transformer Encoder": TransformerPoolingNet(num_layers, input_dim).to(device),
        "Bi-GRU + Attention": BiGRUPoolingNet(num_layers, input_dim).to(device),
        "NetVLAD": NetVLADPoolingNet(num_layers, input_dim).to(device)
    }
    
    results = []
    
    for name, model in models_to_test.items():
        torch.manual_seed(42)
        np.random.seed(42)
        acc, f1, auc, sens, spec = run_experiment(
            name, model, train_loader, test_loader, class_weights, device, epochs=args.epochs
        )
        results.append({
            "Dataset": args.dataset,
            "SSL_Model": args.model,
            "Architecture": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "ROC AUC": auc,
            "Sensitivity (MDD)": sens,
            "Specificity (HC)": spec
        })
        
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    out_csv = f"output/pooling_benchmark_{args.dataset}_{args.model}.csv"
    df.to_csv(out_csv, index=False)
    
    print("=======================================================================================")
    print(f"         TEMPORAL BENCHMARK RESULTS (Dataset: {args.dataset.upper()} | Model: {args.model})")
    print("=======================================================================================")
    print(f"{'Architecture':<22} | {'Accuracy':<8} | {'F1 Score':<8} | {'ROC AUC':<8} | {'Sens (MDD)':<10} | {'Spec (HC)':<10}")
    print("-" * 87)
    for _, row in df.iterrows():
        print(f"{row['Architecture']:<22} | {row['Accuracy']:.4f}   | {row['F1 Score']:.4f}   | {row['ROC AUC']:.4f}   | {row['Sensitivity (MDD)']:.4f}     | {row['Specificity (HC)']:.4f}")
    print("=======================================================================================\n")
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()
