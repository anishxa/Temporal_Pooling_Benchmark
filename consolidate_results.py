import os
import pandas as pd

def main():
    output_dir = "output"
    files = [f for f in os.listdir(output_dir) if f.startswith("pooling_benchmark_") and f.endswith(".csv")]
    if not files:
        print("No benchmark result CSVs found.")
        return
        
    all_dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(output_dir, file))
        all_dfs.append(df)
        
    consolidated = pd.concat(all_dfs, ignore_index=True)
    consolidated.to_csv(os.path.join(output_dir, "temporal_pooling_all_results.csv"), index=False)
    
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
                    f.write(f"| {row['Architecture']} | {row['Accuracy']:.4f} | {row['F1 Score']:.4f} | {row['ROC AUC']:.4f} | {row['Sensitivity (MDD)']:.4f} | {row['Specificity (HC)']:.4f} |\n")
                f.write("\n")
                
    print("Consolidated results saved to output/temporal_pooling_all_results.csv and output/temporal_pooling_summary.md")

if __name__ == "__main__":
    main()
