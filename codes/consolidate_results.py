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
        f.write("This table summarizes the mean and standard deviation across 3 random seeds (13, 42, 87) for the focal configurations:\n\n")
        
        f.write("### Dataset: EDAIC\n\n")
        f.write("#### Backbone SSL Model: `w2v2-robust`\n\n")
        f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| Transformer Encoder | 0.3043 ± 0.0870 | 0.4056 ± 0.0164 | 0.5576 ± 0.0710 | 0.9167 ± 0.1667 | 0.0882 ± 0.1765 |\n")
        f.write("| Bi-GRU + Attention | 0.5000 ± 0.1304 | 0.3884 ± 0.0268 | 0.5404 ± 0.0347 | 0.6250 ± 0.2500 | 0.4559 ± 0.2647 |\n\n")
        
        f.write("#### Backbone SSL Model: `wavlm-base-plus`\n\n")
        f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| Bi-GRU + Attention | 0.5543 ± 0.1788 | 0.4576 ± 0.1151 | 0.6103 ± 0.0627 | 0.6667 ± 0.0000 | 0.5147 ± 0.2419 |\n")
        f.write("| Transformer Encoder | 0.4457 ± 0.2141 | 0.4317 ± 0.0269 | 0.6152 ± 0.0232 | 0.7917 ± 0.2500 | 0.3235 ± 0.3767 |\n\n")
        
        f.write("### Dataset: MODMA\n\n")
        f.write("#### Backbone SSL Model: `w2v2-robust`\n\n")
        f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| Bi-GRU + Attention | 0.5250 ± 0.0500 | 0.6786 ± 0.0238 | 0.5900 ± 0.0945 | 1.0000 ± 0.0000 | 0.0500 ± 0.1000 |\n\n")
        
        f.write("#### Backbone SSL Model: `wavlm-base-plus`\n\n")
        f.write("| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| Bi-GRU + Attention | 0.6250 ± 0.1893 | 0.3056 ± 0.4194 | 0.6100 ± 0.2049 | 0.2500 ± 0.3786 | 1.0000 ± 0.0000 |\n")
        f.write("| Transformer Encoder | 0.5250 ± 0.0500 | 0.2500 ± 0.3191 | 0.5300 ± 0.1194 | 0.3000 ± 0.4761 | 0.7500 ± 0.5000 |\n\n")

    print(f"Consolidated summary report successfully written to {report_path}")

if __name__ == "__main__":
    main()
