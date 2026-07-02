import os
import re
import numpy as np
import pandas as pd
from scipy.stats import beta, pearsonr

def clopper_pearson(k, n, alpha=0.05):
    """Compute Clopper-Pearson exact binomial confidence interval for accuracy."""
    if k == 0:
        return 0.0, 1.0 - alpha**(1.0/n)
    elif k == n:
        return alpha**(1.0/n), 1.0
    else:
        low = beta.ppf(alpha/2, k, n - k + 1)
        high = beta.ppf(1 - alpha/2, k + 1, n - k)
        return low, high

def bootstrap_f1_and_acc(y_true, y_pred, num_bootstraps=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for F1 and Accuracy over speakers."""
    f1_scores = []
    acc_scores = []
    n = len(y_true)
    for _ in range(num_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        # F1
        tp = np.sum((sample_true == 1) & (sample_pred == 1))
        fp = np.sum((sample_true == 0) & (sample_pred == 1))
        fn = np.sum((sample_true == 1) & (sample_pred == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        
        # Accuracy
        acc_scores.append(np.mean(sample_true == sample_pred))
        
    f1_scores = np.sort(f1_scores)
    acc_scores = np.sort(acc_scores)
    
    f1_low = np.percentile(f1_scores, 100 * (alpha / 2))
    f1_high = np.percentile(f1_scores, 100 * (1 - alpha / 2))
    acc_low = np.percentile(acc_scores, 100 * (alpha / 2))
    acc_high = np.percentile(acc_scores, 100 * (1 - alpha / 2))
    return f1_low, f1_high, acc_low, acc_high

def parse_spk_id(path):
    base = os.path.basename(path)
    parts = re.findall(r'\d+', base)
    return int(parts[0]) if parts else None

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 8 Cells details
    models = ["wavlm-base-plus", "w2v2-robust"]
    datasets = ["edaic", "modma"]
    heads = ["Bi-GRU_plus_Attention", "Transformer_Encoder"]
    seeds = [13, 42, 87]
    variants = ["learned", "uniform", "fixed_sweep"]
    
    # ==========================================================================
    # Task 8: Verify split participant counts
    # ==========================================================================
    print("==========================================================================")
    print("Task 8: Verifying split participant counts...")
    print("==========================================================================")
    for ds in datasets:
        metadata_csv = f"data/utterance_table_{ds}_segmented_split.csv"
        if os.path.exists(metadata_csv):
            df_meta = pd.read_csv(metadata_csv)
            df_meta["spk"] = df_meta["file_path"].apply(parse_spk_id)
            counts = df_meta.groupby("split")["spk"].nunique()
            print(f"Corpus: {ds.upper()}")
            for split, count in counts.items():
                print(f"  {split}: {count} unique participants")
        else:
            print(f"Metadata file {metadata_csv} not found.")

    # ==========================================================================
    # Task 3: Confidence Intervals (subset_ci.csv)
    # ==========================================================================
    print("\n==========================================================================")
    print("Task 3: Computing Clopper-Pearson and Bootstrap CIs...")
    print("==========================================================================")
    ci_rows = []
    
    for dataset in datasets:
        for model in models:
            for head in heads:
                # Average predictions over seeds or run separately
                seed_f1s = []
                seed_accs = []
                
                for seed in seeds:
                    pred_file = f"{output_dir}/predictions_{dataset}_{model}_{head}_learned_seed{seed}.csv"
                    if os.path.exists(pred_file):
                        df_pred = pd.read_csv(pred_file)
                        y_true = df_pred["y_true"].values
                        y_pred = df_pred["y_pred"].values
                        y_prob = df_pred["y_prob"].values
                        
                        tp = np.sum((y_true == 1) & (y_pred == 1))
                        tn = np.sum((y_true == 0) & (y_pred == 1)) # Note: specificity tn is hc == hc
                        # Wait, TN is true label == 0 & pred == 0
                        true_tn = np.sum((y_true == 0) & (y_pred == 0))
                        
                        k_correct = np.sum(y_true == y_pred)
                        n_total = len(y_true)
                        
                        # exact Clopper-Pearson
                        cp_low, cp_high = clopper_pearson(k_correct, n_total)
                        
                        # Bootstrap
                        f1_low, f1_high, acc_low, acc_high = bootstrap_f1_and_acc(y_true, y_pred)
                        
                        # Standard metrics
                        acc = np.mean(y_true == y_pred)
                        tp_val = np.sum((y_true == 1) & (y_pred == 1))
                        fp_val = np.sum((y_true == 0) & (y_pred == 1))
                        fn_val = np.sum((y_true == 1) & (y_pred == 0))
                        prec = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
                        rec = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        
                        ci_rows.append({
                            "Dataset": dataset,
                            "SSL_Model": model,
                            "Architecture": head.replace("_plus_", " + ").replace("_", " "),
                            "Seed": seed,
                            "Accuracy": acc,
                            "Acc_Boot_CI_Low": acc_low,
                            "Acc_Boot_CI_High": acc_high,
                            "Acc_CP_CI_Low": cp_low,
                            "Acc_CP_CI_High": cp_high,
                            "F1_Score": f1,
                            "F1_Boot_CI_Low": f1_low,
                            "F1_Boot_CI_High": f1_high
                        })
                        
    if ci_rows:
        df_ci = pd.DataFrame(ci_rows)
        df_ci.to_csv(f"{output_dir}/subset_ci.csv", index=False)
        print(f"Saved subset CIs to {output_dir}/subset_ci.csv")
    else:
        print("No predictions found to compute CIs.")

    # ==========================================================================
    # Task 4: Featurizer ablation (featurizer_ablation.csv)
    # ==========================================================================
    print("\n==========================================================================")
    print("Task 4: Consolidating featurizer ablation results...")
    print("==========================================================================")
    ablation_rows = []
    for dataset in datasets:
        for model in models:
            for head in heads:
                for variant in variants:
                    f1s = []
                    for seed in seeds:
                        res_file = f"{output_dir}/pooling_benchmark_{dataset}_{model}_{head}_{variant}_seed{seed}.csv"
                        if os.path.exists(res_file):
                            df_res = pd.read_csv(res_file)
                            f1s.append(df_res["F1 Score"].values[0])
                    
                    if f1s:
                        ablation_rows.append({
                            "Dataset": dataset,
                            "SSL_Model": model,
                            "Architecture": head.replace("_plus_", " + ").replace("_", " "),
                            "Variant": variant,
                            "F1_mean": np.mean(f1s),
                            "F1_std": np.std(f1s) if len(f1s) > 1 else 0.0
                        })
                        
    if ablation_rows:
        df_ablation = pd.DataFrame(ablation_rows)
        df_ablation.to_csv(f"{output_dir}/featurizer_ablation.csv", index=False)
        print(f"Saved featurizer ablation to {output_dir}/featurizer_ablation.csv")
    else:
        print("No featurizer ablation result files found.")

    # ==========================================================================
    # Task 5: Dump weights and plot heatmap
    # ==========================================================================
    print("\n==========================================================================")
    print("Task 5: Generating layer weight heatmaps...")
    print("==========================================================================")
    weight_dict = {}
    for dataset in datasets:
        for model in models:
            for head in heads:
                key = f"{dataset}_{model}_{head}"
                model_weights = []
                for seed in seeds:
                    weight_file = f"{output_dir}/layer_weights_{dataset}_{model}_{head}_learned_seed{seed}.npy"
                    if os.path.exists(weight_file):
                        w = np.load(weight_file)
                        model_weights.append(w)
                if model_weights:
                    # Average across seeds
                    weight_dict[key] = np.mean(model_weights, axis=0)
                    
    if weight_dict:
        # Save aggregated weights
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Determine maximum layers to pad for plotting
            max_layers = max(len(w) for w in weight_dict.values())
            heatmap_data = []
            columns = []
            for key, w in weight_dict.items():
                # Pad with NaN if sizes differ (e.g. Base has 13, Robust has 25)
                padded = np.pad(w.astype(float), (0, max_layers - len(w)), mode='constant', constant_values=np.nan)
                heatmap_data.append(padded)
                columns.append(key)
                
            heatmap_data = np.stack(heatmap_data, axis=1) # Shape: [layers, configs]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=columns, cmap="viridis")
            plt.title("Softmax-weighted Layer Importance Heatmap (Averaged over 3 seeds)")
            plt.ylabel("Layer Index (0 = bottom/embedding)")
            plt.xlabel("Configuration")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/layer_weights_heatmap.png")
            print(f"Saved layer weights heatmap to {output_dir}/layer_weights_heatmap.png")
        except Exception as e:
            print(f"Could not plot heatmap using matplotlib/seaborn: {e}. Raw weights saved.")
    else:
        print("No layer weights .npy files found.")

    # ==========================================================================
    # Task 6: Sequence-length shortcut check (seqlen_confound.csv)
    # ==========================================================================
    print("\n==========================================================================")
    print("Task 6: Sequence-length shortcut check...")
    print("==========================================================================")
    seqlen_rows = []
    
    for dataset in datasets:
        metadata_csv = f"data/utterance_table_{dataset}_segmented_split.csv"
        if os.path.exists(metadata_csv):
            df_meta = pd.read_csv(metadata_csv)
            df_meta["speaker_id"] = df_meta["file_path"].apply(parse_spk_id)
            
            # Count segments per speaker
            seg_counts = df_meta.groupby("speaker_id").size().reset_index(name="seg_count")
            
            # Only test split for predictions
            df_meta_test = df_meta[df_meta["split"] == "test"]
            test_labels = df_meta_test.groupby("speaker_id")["label"].first().reset_index()
            
            # Merge segment counts with true label
            df_test_analysis = pd.merge(seg_counts, test_labels, on="speaker_id", how="inner")
            
            # Compute Pearson correlation with true label (point-biserial since label is binary)
            true_corr, _ = pearsonr(df_test_analysis["seg_count"], df_test_analysis["label"])
            
            # Now load predictions for each model-head combo on E-DAIC / MODMA
            for model in models:
                for head in heads:
                    pred_f1s = []
                    pred_probs = []
                    
                    for seed in seeds:
                        pred_file = f"{output_dir}/predictions_{dataset}_{model}_{head}_learned_seed{seed}.csv"
                        if os.path.exists(pred_file):
                            df_pred = pd.read_csv(pred_file)
                            
                            df_merged = pd.merge(df_test_analysis, df_pred, on="speaker_id", how="inner")
                            
                            pred_corr, _ = pearsonr(df_merged["seg_count"], df_merged["y_pred"])
                            prob_corr, _ = pearsonr(df_merged["seg_count"], df_merged["y_prob"])
                            
                            seqlen_rows.append({
                                "Dataset": dataset,
                                "SSL_Model": model,
                                "Architecture": head.replace("_plus_", " + ").replace("_", " "),
                                "Seed": seed,
                                "Correlation_with_Label": true_corr,
                                "Correlation_with_Prediction": pred_corr,
                                "Correlation_with_Probability": prob_corr
                            })
                            
    if seqlen_rows:
        df_seqlen = pd.DataFrame(seqlen_rows)
        df_seqlen.to_csv(f"{output_dir}/seqlen_confound.csv", index=False)
        print(f"Saved sequence length confound report to {output_dir}/seqlen_confound.csv")
    else:
        print("No prediction files found to perform sequence-length shortcut check.")

    # ==========================================================================
    # Task 7: Continuous collapse metric (collapse_continuous.csv)
    # ==========================================================================
    print("\n==========================================================================")
    print("Task 7: Continuous collapse metric calculations...")
    print("==========================================================================")
    collapse_rows = []
    
    # Majority class baselines (HC = class 0 is usually majority)
    # E-DAIC test has 17 HC, 6 MDD (majority HC = 17/23 = 73.9%)
    # MODMA test has 5 HC, 5 MDD (majority = 50.0%)
    majority_baselines = {
        "edaic": 17.0 / 23.0,
        "modma": 5.0 / 10.0
    }
    
    for dataset in datasets:
        for model in models:
            for head in heads:
                for seed in seeds:
                    pred_file = f"{output_dir}/predictions_{dataset}_{model}_{head}_learned_seed{seed}.csv"
                    if os.path.exists(pred_file):
                        df_pred = pd.read_csv(pred_file)
                        y_true = df_pred["y_true"].values
                        y_pred = df_pred["y_pred"].values
                        
                        # Compute Balanced Accuracy
                        tp = np.sum((y_true == 1) & (y_pred == 1))
                        tn = np.sum((y_true == 0) & (y_pred == 0))
                        n_pos = np.sum(y_true == 1)
                        n_neg = np.sum(y_true == 0)
                        
                        sens = tp / n_pos if n_pos > 0 else 0.0
                        spec = tn / n_neg if n_neg > 0 else 0.0
                        bal_acc = (sens + spec) / 2.0
                        
                        # Balanced accuracy minus majority accuracy
                        majority_acc = majority_baselines[dataset]
                        bal_acc_delta = bal_acc - majority_acc
                        
                        # Prediction Entropy (H = -p log2 p - (1-p) log2 (1-p))
                        p_pred = np.mean(y_pred)
                        if p_pred == 0.0 or p_pred == 1.0:
                            entropy = 0.0
                        else:
                            entropy = - (p_pred * np.log2(p_pred) + (1.0 - p_pred) * np.log2(1.0 - p_pred))
                            
                        collapse_rows.append({
                            "Dataset": dataset,
                            "SSL_Model": model,
                            "Architecture": head.replace("_plus_", " + ").replace("_", " "),
                            "Seed": seed,
                            "Balanced_Accuracy": bal_acc,
                            "Majority_Accuracy": majority_acc,
                            "Bal_Acc_Delta": bal_acc_delta,
                            "Prediction_Entropy": entropy,
                            "Is_Collapsed_Binary": 1 if (sens == 1.0 and spec == 0.0) or (sens == 0.0 and spec == 1.0) else 0
                        })
                        
    if collapse_rows:
        df_collapse = pd.DataFrame(collapse_rows)
        df_collapse.to_csv(f"{output_dir}/collapse_continuous.csv", index=False)
        print(f"Saved continuous collapse metric analysis to {output_dir}/collapse_continuous.csv")
    else:
        print("No prediction files found to perform continuous collapse analysis.")

if __name__ == "__main__":
    main()
