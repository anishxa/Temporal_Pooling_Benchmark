import os
import pandas as pd
import numpy as np

def main():
    output_dir = "output"
    all_files = [
        f for f in os.listdir(output_dir) 
        if f.startswith("pooling_benchmark_") and f.endswith(".csv") 
        and f != "pooling_benchmark_results.csv" 
        and f != "temporal_pooling_all_results.csv"
        and f != "pooling_benchmark_statistical_CIs.csv"
        and f != "featurizer_ablation.csv"
        and f != "collapse_continuous.csv"
        and f != "seqlen_confound.csv"
        and f != "subset_ci.csv"
    ]
    if not all_files:
        print("No benchmark result CSVs found.")
        return

    # 1. Compile the Primary Grid (Single-Seed Run)
    # The original single-run files do not contain "_seed" in their filename
    single_files = [f for f in all_files if "_seed" not in f]
    single_dfs = []
    for file in single_files:
        df = pd.read_csv(os.path.join(output_dir, file))
        if "Featurizer_Type" not in df.columns:
            df["Featurizer_Type"] = "learned"
        single_dfs.append(df)
    
    if single_dfs:
        single_raw = pd.concat(single_dfs, ignore_index=True)
        single_grid = single_raw[single_raw["Featurizer_Type"] == "learned"].copy()
        single_grid.to_csv(os.path.join(output_dir, "temporal_pooling_all_results.csv"), index=False)
        print(f"Consolidated single-seed results saved to output/temporal_pooling_all_results.csv")
    else:
        single_grid = None
        print("Warning: No single-seed files found.")

    # 2. Generate Markdown Summary Report containing BOTH tables
    report_path = os.path.join(output_dir, "temporal_pooling_summary.md")
    with open(report_path, "w") as f:
        f.write("# DEPOOL: Expanded Comparison Report\n\n")
        f.write("This report summarizes the pooling architecture comparison across 6 speech SSL backbones on E-DAIC and MODMA.\n\n")
        
        # Section A: Primary Comparison Grid (Single-Seed Run, Seed 42)
        f.write("## 1. Primary Comparison Grid (Single-Seed, matching Tables II & III in Paper)\n\n")
        if single_grid is not None:
            for dataset in sorted(single_grid["Dataset"].unique()):
                f.write(f"### Dataset: {dataset.upper()}\n\n")
                df_ds = single_grid[single_grid["Dataset"] == dataset]
                
                for model in sorted(df_ds["SSL_Model"].unique()):
                    f.write(f"#### Backbone SSL Model: `{model}`\n\n")
                    df_model = df_ds[df_ds["SSL_Model"] == model].copy()
                    df_model = df_model.sort_values(by="F1 Score", ascending=False)
                    
                    f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
                    f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                    for _, row in df_model.iterrows():
                        f.write(f"| {row['Architecture']} | {row['Accuracy']:.4f} | {row['F1 Score']:.4f} | {row['ROC AUC']:.4f} | {row['Sensitivity (MDD)']:.4f} | {row['Specificity (HC)']:.4f} |\n")
                    f.write("\n")
        else:
            f.write("No single-seed run results found.\n\n")

        # Section B: Seed Sensitivity Analysis (3-Seed Robustness, matching Table IV in Paper)
        f.write("\n## 2. Seed Sensitivity Analysis (Multi-Seed Averages, matching Table IV in Paper)\n\n")
        f.write("This table summarizes the mean and standard deviation across 3 random seeds (13, 42, 87) for the focal configurations, computed dynamically from authentic model prediction files:\n\n")
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

        datasets = ["edaic", "modma"]
        models = ["wavlm-base-plus", "w2v2-robust"]
        heads = ["Bi-GRU + Attention", "Transformer Encoder"]
        seeds = [13, 42, 87]

        for ds in datasets:
            f.write(f"### Dataset: {ds.upper()}\n\n")
            for model in models:
                f.write(f"#### Backbone SSL Model: `{model}`\n\n")
                f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                
                for head in heads:
                    head_safe = head.replace(" ", "_").replace("+", "plus")
                    accs, f1s, aucs, sens_list, spec_list = [], [], [], [], []
                    
                    for seed in seeds:
                        pred_file = f"{output_dir}/predictions_{ds}_{model}_{head_safe}_learned_seed{seed}.csv"
                        if os.path.exists(pred_file):
                            df_p = pd.read_csv(pred_file)
                            yt = df_p["y_true"].values
                            yp = df_p["y_pred"].values
                            yb = df_p["y_prob"].values
                            
                            acc = accuracy_score(yt, yp)
                            f1 = f1_score(yt, yp, zero_division=0)
                            try:
                                auc = roc_auc_score(yt, yb)
                            except:
                                auc = 0.5
                                
                            cm = confusion_matrix(yt, yp, labels=[0, 1])
                            sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
                            spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
                            
                            accs.append(acc)
                            f1s.append(f1)
                            aucs.append(auc)
                            sens_list.append(sens)
                            spec_list.append(spec)
                    
                    if f1s:
                        f.write(f"| {head} | {np.mean(accs):.4f} ± {np.std(accs):.4f} | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f} | {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f} |\n")
                f.write("\n")

    print(f"Consolidated summary report successfully written to {report_path}")

if __name__ == "__main__":
    main()
