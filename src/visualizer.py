import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np

def extract_quantization(model_name):
    match = re.search(r'(q\d[a-z0-9_]*)\.gguf', model_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "Unknown"

def extract_model_size(model_name):
    # Extract parameter size like 1.5b, 3b, 7b, 8b
    match = re.search(r'([\d\.]+)(b)', model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 7.0 # Default fallback if not found

def create_dashboard():
    # Adjust paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "../results")
    csv_file = os.path.join(results_dir, "production_benchmarks.csv")
    thermal_csv_file = os.path.join(results_dir, "thermal_log.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    # Calculate 95% CI for TPS per model/context
    # CI = 1.96 * (std / sqrt(n))
    ci_df = df.groupby(['model', 'context_length']).agg(
        std_tps=('tokens_per_sec', 'std'),
        count=('tokens_per_sec', 'count')
    ).reset_index()
    
    ci_df['tps_ci'] = 1.96 * (ci_df['std_tps'] / np.sqrt(ci_df['count']))
    
    # Merge CI back into the main dataframe
    df = pd.merge(df, ci_df[['model', 'context_length', 'tps_ci']], on=['model', 'context_length'], how='left')

    df['quantization'] = df['model'].apply(extract_quantization)
    df['model_size_b'] = df['model'].apply(extract_model_size)

    # Aesthetics
    sns.set_theme(style="whitegrid", rc={"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.titlesize": 14})
    sns.set_palette('magma')

    # Creating subplots: 3x2 grid to accommodate the new charts
    fig, axes = plt.subplots(3, 2, figsize=(24, 18))
    
    # Filter dataset for the standard 2048 context length for the main plots
    df_2048 = df[df['context_length'] == 2048]
    
    # 1. Throughput Plot with calculated error bars
    # Since seaborn's barplot automatically calculates CI, we can use it directly, 
    # but the user requested explicit representation. Seaborn handles ci=95 by default on multiple rows.
    sns.barplot(data=df_2048, x='model', y='tokens_per_sec', hue='family', errorbar=('ci', 95), capsize=.1, ax=axes[0, 0])
    axes[0, 0].set_title('Throughput with 95% CI (Tokens/Sec) [Context: 2048]')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), ha='right')

    # 2. VRAM Plot
    sns.barplot(data=df_2048, x='model', y='max_vram_mb', hue='family', errorbar=('ci', 95), capsize=.1, ax=axes[0, 1])
    axes[0, 1].set_title('Max VRAM with 95% CI (MB) [Context: 2048]')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), ha='right')

    # 3. Energy Efficiency Plot
    if 'tokens_per_joule' in df_2048.columns:
        sns.barplot(data=df_2048, x='model', y='tokens_per_joule', hue='family', errorbar=('ci', 95), capsize=.1, ax=axes[1, 0])
        axes[1, 0].set_title('Energy Efficiency (Tokens/Joule) [Context: 2048]')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), ha='right')
    else:
        axes[1, 0].axis('off')

    # 4. VRAM Usage vs. Context Window
    context_models = df[df['context_length'].isin([512, 8192])]['model'].unique()
    context_df = df[df['model'].isin(context_models)]
    if not context_df.empty:
        sns.lineplot(data=context_df, x='context_length', y='max_vram_mb', hue='model', marker='o', errorbar=('ci', 95), ax=axes[1, 1])
        axes[1, 1].set_title('VRAM Usage vs. Context Window (95% CI)')
        axes[1, 1].set_xticks([512, 2048, 8192])
        axes[1, 1].set_xlabel('Context Length (Tokens)')
        axes[1, 1].set_ylabel('Max VRAM (MB)')
        
    # 5. The Efficiency Frontier: Perplexity vs. Tokens per Joule
    if 'perplexity' in df_2048.columns and 'tokens_per_joule' in df_2048.columns:
        # Filter out failed perplexity runs (0.0)
        frontier_df = df_2048[df_2048['perplexity'] > 0.0].groupby(['model', 'family', 'quantization', 'model_size_b']).mean(numeric_only=True).reset_index()
        
        # Sizing by model_size_b requires mapping it to a reasonable visual scale
        size_scale = frontier_df['model_size_b'] * 100
        
        sns.scatterplot(data=frontier_df, x='perplexity', y='tokens_per_joule', hue='family', size='model_size_b', sizes=(100, 500), ax=axes[2, 0])
        axes[2, 0].set_title('The Efficiency Frontier: Perplexity vs. Energy')
        axes[2, 0].set_xlabel('WikiText-2 Perplexity (Lower is Better)')
        axes[2, 0].set_ylabel('Energy Efficiency (Tokens / Joule)')
        
        for i in range(len(frontier_df)):
            x_val = frontier_df['perplexity'].iloc[i]
            y_val = frontier_df['tokens_per_joule'].iloc[i]
            label = f"{frontier_df['model_size_b'].iloc[i]}B {frontier_df['quantization'].iloc[i]}"
            axes[2, 0].text(x_val + 0.1, y_val, label, horizontalalignment='left', size=10, color='black')
    else:
         axes[2, 0].axis('off')

    # 6. Thermal Decay Logic: Multi-axis Line Plot
    if os.path.exists(thermal_csv_file):
        try:
            tdf = pd.read_csv(thermal_csv_file)
            if not tdf.empty:
                # Normalize timestamp to start at 0
                tdf['time_sec'] = tdf['timestamp'] - tdf['timestamp'].min()
                
                # Plot Temperature
                sns.lineplot(data=tdf, x='time_sec', y='temp_c', color='red', label='Temp (°C)', alpha=0.7, ax=axes[2, 1])
                axes[2, 1].set_ylabel('Temperature (°C)', color='red')
                axes[2, 1].tick_params(axis='y', labelcolor='red')
                
                # Create a twin axis for Clock Speed
                ax2 = axes[2, 1].twinx()
                sns.lineplot(data=tdf, x='time_sec', y='clock_mhz', color='blue', label='Clock (MHz)', alpha=0.7, ax=ax2)
                ax2.set_ylabel('SM Clock Speed (MHz)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                
                # Highlight when clock drops below base clock (assuming ~1440 MHz for RTX 3080 base)
                base_clock = 1440
                throttled = tdf[tdf['clock_mhz'] < base_clock]
                if not throttled.empty:
                    ax2.scatter(throttled['time_sec'], throttled['clock_mhz'], color='black', marker='x', s=50, label='Sub-Base Clock', zorder=5)
                
                # Consolidate legends
                lines_1, labels_1 = axes[2, 1].get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')
                axes[2, 1].get_legend().remove() # Remove the original seaborn legend

                axes[2, 1].set_title('Thermal Decay & Throttling Analysis')
                axes[2, 1].set_xlabel('Continuous Runtime (Seconds)')
            else:
                 axes[2, 1].axis('off')
        except pd.errors.EmptyDataError:
             axes[2, 1].axis('off')
    else:
        axes[2, 1].axis('off')

    plt.tight_layout()
    output_img = os.path.join(results_dir, "dashboard.png")
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    print(f"Scientific visualization saved to {output_img}")

if __name__ == "__main__":
    create_dashboard()