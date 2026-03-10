import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def extract_quantization(model_name):
    # Try to extract common quantization patterns like Q4_K_M, q8_0, etc.
    match = re.search(r'(q\d[a-z0-9_]*)\.gguf', model_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "Unknown"

def create_dashboard():
    # Adjust paths for the new /src structure
    base_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "../results")
    csv_file = os.path.join(results_dir, "production_benchmarks.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    print("\n--- LLM Inference Benchmark Dashboard ---\n")
    print(df.to_string(index=False))
    print("\n-----------------------------------------\n")

    # Add quantization column for scatter plot labels
    df['quantization'] = df['model'].apply(extract_quantization)

    # Set up Seaborn style
    sns.set_theme(style="whitegrid")
    palette = "viridis"

    # Creating subplots: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # Filter dataset for the standard 2048 context length for the main plots
    df_2048 = df[df['context_length'] == 2048]
    
    # 1. Latency Plot
    # Seaborn automatically aggregates multiple runs and draws error bars (95% CI by default)
    sns.barplot(data=df_2048, x='model', y='latency_sec', hue='family', palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title('Latency (Seconds) [Context: 2048]')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), ha='right')

    # 2. Throughput Plot
    sns.barplot(data=df_2048, x='model', y='tokens_per_sec', hue='family', palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title('Throughput (Tokens/Sec) [Context: 2048]')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), ha='right')

    # 3. VRAM Plot
    sns.barplot(data=df_2048, x='model', y='max_vram_mb', hue='family', palette=palette, ax=axes[0, 2])
    axes[0, 2].set_title('Max VRAM (MB) [Context: 2048]')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), ha='right')

    # 4. Scatter Plot: Throughput vs. VRAM
    # Aggregate data to plot a single point per model/quantization (mean of the 3 runs)
    agg_df = df_2048.groupby(['model', 'family', 'quantization']).mean(numeric_only=True).reset_index()
    sns.scatterplot(data=agg_df, x='max_vram_mb', y='tokens_per_sec', hue='family', palette=palette, s=150, ax=axes[1, 0])
    axes[1, 0].set_title('Throughput vs. Max VRAM [Context: 2048]')
    
    # Label the scatter points with quantization
    for i in range(len(agg_df)):
        x_val = agg_df['max_vram_mb'].iloc[i]
        y_val = agg_df['tokens_per_sec'].iloc[i]
        label = agg_df['quantization'].iloc[i]
        axes[1, 0].text(x_val + 50, y_val, label, horizontalalignment='left', size='medium', color='black', weight='semibold')

    # 5. VRAM Usage vs. Context Window (New Chart)
    # Filter data to only include the model tested across multiple context windows
    context_models = df[df['context_length'].isin([512, 8192])]['model'].unique()
    context_df = df[df['model'].isin(context_models)]
    if not context_df.empty:
        # Error bars will show the std deviation across the 3 runs for each context length
        sns.lineplot(data=context_df, x='context_length', y='max_vram_mb', hue='model', marker='o', err_style="bars", ax=axes[1, 1])
        axes[1, 1].set_title('VRAM Usage vs. Context Window')
        axes[1, 1].set_xticks([512, 2048, 8192])
        axes[1, 1].set_xlabel('Context Length (Tokens)')
        axes[1, 1].set_ylabel('Max VRAM (MB)')
        
    # Hide the empty subplot at [1, 2]
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_img = os.path.join(results_dir, "dashboard.png")
    plt.savefig(output_img, bbox_inches='tight')
    print(f"Dashboard visualization saved to {output_img}")

if __name__ == "__main__":
    create_dashboard()