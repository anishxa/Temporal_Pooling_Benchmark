import os
import pandas as pd
import numpy as np

def main():
    output_dir = "output"
    files = [
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
    if not files:
        print("No benchmark result CSVs found.")
        return
        
    all_dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(output_dir, file))
        # Ensure Featurizer_Type is present
        if "Featurizer_Type" not in df.columns:
            df["Featurizer_Type"] = "learned"
        all_dfs.append(df)
        
    raw_df = pd.concat(all_dfs, ignore_index=True)
    
    # We only include the 'learned' variant in the main paper's table (the 72 grid)
    main_grid_df = raw_df[raw_df["Featurizer_Type"] == "learned"].copy()
    
    # Group by Dataset, SSL_Model, Architecture and calculate mean and std
    grouped = main_grid_df.groupby(["Dataset", "SSL_Model", "Architecture"])
    
    consolidated_rows = []
    for (dataset, model, arch), group in grouped:
        means = group[["Accuracy", "F1 Score", "ROC AUC", "Sensitivity (MDD)", "Specificity (HC)"]].mean()
        # fillna(0) for std since single seed runs will have NaN std
        stds = group[["Accuracy", "F1 Score", "ROC AUC", "Sensitivity (MDD)", "Specificity (HC)"]].std().fillna(0)
        
        row = {
            "Dataset": dataset,
            "SSL_Model": model,
            "Architecture": arch,
            "Accuracy": means["Accuracy"],
            "F1 Score": means["F1 Score"],
            "ROC AUC": means["ROC AUC"],
            "Sensitivity (MDD)": means["Sensitivity (MDD)"],
            "Specificity (HC)": means["Specificity (HC)"],
            "Accuracy_std": stds["Accuracy"],
            "F1_std": stds["F1 Score"],
            "ROC_AUC_std": stds["ROC AUC"],
            "Sensitivity_std": stds["Sensitivity (MDD)"],
            "Specificity_std": stds["Specificity (HC)"]
        }
        consolidated_rows.append(row)
        
    consolidated = pd.DataFrame(consolidated_rows)
    
    # Assert that after grouping we have exactly 72 configurations (the full grid)
    if len(consolidated) != 72:
        raise ValueError(f"Expected exactly 72 configurations after grouping, but found {len(consolidated)}. Check if some baseline files are missing.")
        
    consolidated.to_csv(os.path.join(output_dir, "temporal_pooling_all_results.csv"), index=False)
    print(f"Consolidated results saved to output/temporal_pooling_all_results.csv")
    
    # Generate Markdown summary
    with open(os.path.join(output_dir, "temporal_pooling_summary.md"), "w") as f:
        f.write("# Temporal Pooling Benchmark: Expanded Comparison Report\n\n")
        f.write("This report compares 6 pooling architectures across 6 speech Self-Supervised Learning (SSL) backbones on **E-DAIC** and **MODMA** datasets using **semi-fine-tuning** (learnable weighted sum of hidden layers).\n\n")
        
        for dataset in sorted(consolidated["Dataset"].unique()):
            f.write(f"## Dataset: {dataset.upper()}\n\n")
            df_ds = consolidated[consolidated["Dataset"] == dataset]
            
            # Sort models
            for model in sorted(df_ds["SSL_Model"].unique()):
                f.write(f"### Backbone SSL Model: `{model}`\n\n")
                df_model = df_ds[df_ds["SSL_Model"] == model].copy()
                df_model = df_model.sort_values(by="F1 Score", ascending=False)
                
                f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                for _, row in df_model.iterrows():
                    # Format with std if std > 0
                    if row["F1_std"] > 0:
                        f.write(f"| {row['Architecture']} | {row['Accuracy']:.4f} ± {row['Accuracy_std']:.4f} | {row['F1 Score']:.4f} ± {row['F1_std']:.4f} | {row['ROC AUC']:.4f} ± {row['ROC_AUC_std']:.4f} | {row['Sensitivity (MDD)']:.4f} ± {row['Sensitivity_std']:.4f} | {row['Specificity (HC)']:.4f} ± {row['Specificity_std']:.4f} |\n")
                    else:
                        f.write(f"| {row['Architecture']} | {row['Accuracy']:.4f} | {row['F1 Score']:.4f} | {row['ROC AUC']:.4f} | {row['Sensitivity (MDD)']:.4f} | {row['Specificity (HC)']:.4f} |\n")
                f.write("\n")
                
    print("Consolidated summary markdown saved to output/temporal_pooling_summary.md")

if __name__ == "__main__":
    main()
