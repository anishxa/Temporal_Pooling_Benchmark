import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from dataset import get_dataloaders
from models import (
    MeanPoolingNet, StatisticalPoolingNet, SelfAttentionPoolingNet, 
    TransformerPoolingNet, BiGRUPoolingNet, NetVLADPoolingNet
)

def normalize_name(name):
    return name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")

def evaluate(model, data_loader, device, return_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_spk_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                seqs, labels, masks, spk_ids = batch
            else:
                seqs, labels, masks = batch
                spk_ids = torch.zeros_like(labels)
                
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            logits = model(seqs, masks)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_spk_ids.extend(spk_ids.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_spk_ids = np.array(all_spk_ids)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    if return_predictions:
        return acc, f1, auc, sensitivity, specificity, all_labels, all_preds, all_probs, all_spk_ids
    return acc, f1, auc, sensitivity, specificity

def run_experiment(model_name, model, train_loader, val_loader, test_loader, class_weights, device, epochs=50):
    print(f"--- Running Experiment for: {model_name} ---")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_f1 = -1.0
    best_state_dict = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if len(batch) == 4:
                seqs, labels, masks, _ = batch
            else:
                seqs, labels, masks = batch
            seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(seqs, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluate on validation set
        val_acc, val_f1, val_auc, val_sens, val_spec = evaluate(model, val_loader, device)
        
        # Checkpoint based on validation F1
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val F1: {val_f1:.4f}")
            
    # Load the best validation checkpoint before evaluating on test set
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    # Evaluate final model on test set exactly once
    test_acc, test_f1, test_auc, test_sens, test_spec, test_labels, test_preds, test_probs, test_spk_ids = evaluate(
        model, test_loader, device, return_predictions=True
    )
    print(f"Test metrics: Acc={test_acc:.4f}, F1={test_f1:.4f}\n")
    return test_acc, test_f1, test_auc, test_sens, test_spec, (test_labels, test_preds, test_probs, test_spk_ids), best_val_f1

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
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Run only a specific pooling head"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Process-level random seed"
    )
    parser.add_argument(
        "--featurizer_type",
        type=str,
        default="learned",
        choices=["learned", "uniform", "fixed", "fixed_sweep"],
        help="Type of featurizer: learned layer weights, uniform mean, fixed single layer, or sweep over all layers to pick best on validation set"
    )
    parser.add_argument(
        "--fixed_layer_idx",
        type=int,
        default=None,
        help="Layer index when featurizer_type is fixed"
    )
    args = parser.parse_args()

    # === Set process-level seed once at start ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Force CPU device for training downstream classifiers (avoids MPS bugs on recurrent layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {args.seed} | Featurizer: {args.featurizer_type}")

    print(f"Loading data for dataset={args.dataset}, model={args.model}...")
    try:
        train_loader, val_loader, test_loader, y_train = get_dataloaders(
            dataset_name=args.dataset,
            model_name=args.model,
            max_len=args.max_len,
            batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(e)
        return
        
    class_counts = torch.bincount(torch.tensor(y_train).long())
    class_weights = len(y_train) / (len(class_counts) * class_counts.float())
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

    # Models classes definitions
    models_classes = {
        "Mean Pooling": lambda **kwargs: MeanPoolingNet(num_layers, input_dim, **kwargs),
        "Statistical Pooling": lambda **kwargs: StatisticalPoolingNet(num_layers, input_dim, **kwargs),
        "Self-Attention": lambda **kwargs: SelfAttentionPoolingNet(num_layers, input_dim, **kwargs),
        "Transformer Encoder": lambda **kwargs: TransformerPoolingNet(num_layers, input_dim, **kwargs),
        "Bi-GRU + Attention": lambda **kwargs: BiGRUPoolingNet(num_layers, input_dim, **kwargs),
        "NetVLAD": lambda **kwargs: NetVLADPoolingNet(num_layers, input_dim, **kwargs)
    }
    
    if args.head:
        norm_head = normalize_name(args.head)
        matched_head = None
        for k in models_classes.keys():
            if normalize_name(k) == norm_head:
                matched_head = k
                break
        if matched_head is None:
            raise ValueError(f"Unknown head: '{args.head}'. Available: {list(models_classes.keys())}")
        models_classes = {matched_head: models_classes[matched_head]}

    results = []
    for name, get_model in models_classes.items():
        print(f"\n==========================================================================")
        print(f"Evaluating {name}...")
        print(f"==========================================================================")
        
        best_layer = -1
        
        if args.featurizer_type == "fixed_sweep":
            best_val_f1 = -1.0
            best_test_metrics = None
            best_preds = None
            best_weights = None
            
            # Sweep layers, pick best on validation set
            for l_idx in range(num_layers):
                # Reset seed for exact identical initialization across layer sweeps
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(args.seed)
                    
                model = get_model(featurizer_type="fixed", fixed_layer_idx=l_idx).to(device)
                acc, f1, auc, sens, spec, preds, val_f1 = run_experiment(
                    f"{name} (Layer {l_idx})", model, train_loader, val_loader, test_loader, class_weights, device, epochs=args.epochs
                )
                if val_f1 > best_val_f1 or best_layer == -1:
                    best_val_f1 = val_f1
                    best_layer = l_idx
                    best_test_metrics = [acc, f1, auc, sens, spec]
                    best_preds = preds
                    best_weights = model.featurizer.last_weights
                    
            print(f"--> Best layer for {name} by Val F1 is Layer {best_layer} (Val F1 = {best_val_f1:.4f})")
            acc, f1, auc, sens, spec = best_test_metrics
            test_preds = best_preds
            last_weights = best_weights
        else:
            model = get_model(featurizer_type=args.featurizer_type, fixed_layer_idx=args.fixed_layer_idx).to(device)
            acc, f1, auc, sens, spec, test_preds, val_f1 = run_experiment(
                name, model, train_loader, val_loader, test_loader, class_weights, device, epochs=args.epochs
            )
            last_weights = model.featurizer.last_weights
            best_layer = args.fixed_layer_idx if args.fixed_layer_idx is not None else -1

        # Save test predictions for speaker-level bootstrapping (CI calculations)
        head_safe = name.replace(" ", "_").replace("+", "plus")
        os.makedirs("output", exist_ok=True)
        pred_csv = f"output/predictions_{args.dataset}_{args.model}_{head_safe}_{args.featurizer_type}_seed{args.seed}.csv"
        labels, preds, probs, spk_ids = test_preds
        pd.DataFrame({
            "speaker_id": spk_ids,
            "y_true": labels,
            "y_pred": preds,
            "y_prob": probs
        }).to_csv(pred_csv, index=False)
        print(f"Saved test predictions to {pred_csv}")

        # Dump learned weights (Task 5)
        if last_weights is not None:
            weights_npy = f"output/layer_weights_{args.dataset}_{args.model}_{head_safe}_{args.featurizer_type}_seed{args.seed}.npy"
            np.save(weights_npy, last_weights.cpu().numpy())
            print(f"Saved layer weights to {weights_npy}")

        results.append({
            "Dataset": args.dataset,
            "SSL_Model": args.model,
            "Architecture": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "ROC AUC": auc,
            "Sensitivity (MDD)": sens,
            "Specificity (HC)": spec,
            "Seed": args.seed,
            "Featurizer_Type": args.featurizer_type,
            "Best_Layer": best_layer
        })
        
    df = pd.DataFrame(results)
    out_csv = f"output/pooling_benchmark_{args.dataset}_{args.model}_{head_safe}_{args.featurizer_type}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved run results to {out_csv}")

if __name__ == "__main__":
    main()
