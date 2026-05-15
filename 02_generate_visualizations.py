# -*- coding: utf-8 -*-
"""
Technical Plotting Script (JD Report).
Version 2.0 with global configuration and 4-phase analysis.
All labels and titles in English.
"""

import argparse
import os
import re
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tabulate import tabulate

try:
    import dataframe_image as dfi
    HAS_DFI = True
except ImportError:
    HAS_DFI = False

# ==========================================
# GLOBAL CONFIGURATION AND CONTROL VARIABLES
# ==========================================
# Modifying these variables will automatically update all plots and tables.

TARGET_METRIC = 'F1_Macro' # Options: 'OA', 'AA', 'F1_Macro', 'Kappa'
TARGET_CL = 'XGB'    # Fixed classifier for Phases 3 and 4 (e.g., 'XGB', 'SVM', 'RF')
TARGET_K = 1.0       # Fixed data proportion for Phase 4 (Appendix)
TARGET_D = 64        # Fixed latent dimension for Phase 4 (Appendix)

# Datasets to include in the multi-plots
DATASETS_ORDER = ['Indian_Pines', 'Pavia_University', 'Salinas', 'KSC']
K_VALUES = [0.25, 1.0]

# Metric mapping from friendly names to CSV columns
METRIC_MAP = {
    'OA': 'tst_oa',
    'AA': 'tst_aa',
    'F1_Macro': 'tst_f1_macro',
    'Kappa': 'tst_kappa'
}

COLOR_MAP = {
    "VIT": "#3498db",      # Bright Blue
    "MAE_VIT": "#e67e22",  # Carrot Orange
    "DINO_VIT": "#2ecc71", # Emerald Green
    "AE_Patch": "#e74c3c", # Alizarin Red
    "AE": "#9b59b6",       # Amethyst Purple
    "PCA": "#34495e"       # Wet Asphalt (Grey)
}

# Specific colors for Classifiers (Phase 2)
CL_COLOR_MAP = {
    'NB': '#ddbc20',
    'DT': '#2596ff',
    'SVM': '#98489c',
    'RF': '#23df3f',
    'XGB': '#e98357'
}

# ==========================================
# DATASET REGISTRY AND METADATA
# ==========================================
DATASET_REGISTRY = {
    "Indian_Pines": {
        "class_stats": {
            'Alfalfa': 46, 'Corn-notill': 1428, 'Corn-mintill': 830, 'Corn': 237,
            'Grass-pasture': 483, 'Grass-trees': 730, 'Grass-pasture-mowed': 28,
            'Hay-windrowed': 478, 'Oats': 20, 'Soybean-notill': 972,
            'Soybean-mintill': 2455, 'Soybean-clean': 593, 'Wheat': 205,
            'Woods': 1265, 'Buildings-Grass-Trees-Drives': 386, 'Stone-Steel-Towers': 93
        }
    },
    "Pavia_University": {
        "class_stats": {
            'Asphalt': 6631, 'Meadows': 18649, 'Gravel': 2099, 'Trees': 3064,
            'Metal Sheets': 1345, 'Bare Soil': 5029, 'Bitumen': 1330,
            'Bricks': 3682, 'Shadows': 947
        }
    },
    "KSC": {
        "class_stats": {
            'Scrub': 761, 'Willow swamp': 243, 'CP hammock': 256, 'Slash pine': 252,
            'Oak/Broadleaf': 161, 'Hardwood': 229, 'Swamp': 105, 'Graminoid marsh': 431,
            'Spartina marsh': 520, 'Cattail marsh': 404, 'Salt marsh': 419,
            'Mud flats': 503, 'Water': 927
        }
    },
    "Salinas": {
        "class_stats": {
            'Brocoli green weeds 1': 12009, 'Brocoli green weeds 2': 3726, 'Fallow': 1976,
            'Fallow rough plow': 1394, 'Fallow smooth': 2678, 'Stubble': 3959,
            'Celery': 3579, 'Grapes untrained': 11271, 'Soil vinyard develop': 6203,
            'Corn senesced green weeds': 3278, 'Lettuce romaine 4wk': 1068,
            'Lettuce romaine 5wk': 1927, 'Lettuce romaine 6wk': 916,
            'Lettuce romaine 7wk': 1070, 'Vinyard untrained': 7268, 'Vinyard vertical trellis': 1807
        }
    }
}

def apply_style():
    """Applies a premium academic visual style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
    })
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ==========================================
# PHASE 1: METRIC DIVERGENCE TABLE
# ==========================================
def generate_phase1_table(df, output_dir):
    print("\n>>> [PHASE 1] Generating Metric Divergence Table (OA vs AA vs F1)...")
    
    # 1. Group by exact configuration and average the k-folds
    group_cols = ['dataset', 'ftr_extractor', 'classifier', 'd', 'k']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metrics_to_mean = [c for c in num_cols if c not in ['d', 'k', 'fold']]
    
    df_mean = df.groupby(group_cols)[metrics_to_mean].mean().reset_index()

    res_list = []
    
    for ds in df_mean['dataset'].unique():
        ds_data = df_mean[df_mean['dataset'] == ds]
        
        # Minority class info
        stats = DATASET_REGISTRY.get(ds, {}).get("class_stats", {})
        if stats:
            min_class = min(stats, key=stats.get)
            min_class_idx = list(stats.keys()).index(min_class)
            min_f1_col = f'tst_f1_c{min_class_idx}'
        else:
            min_class = "N/A"
            min_f1_col = None
            
        # 2. Find absolute winners for each metric
        winner_oa = ds_data.loc[ds_data['tst_oa'].idxmax()]
        winner_aa = ds_data.loc[ds_data['tst_aa'].idxmax()]
        winner_f1 = ds_data.loc[ds_data['tst_f1_macro'].idxmax()]
        
        targets = [('Max OA', winner_oa), ('Max AA', winner_aa), ('Max F1', winner_f1)]
        
        for target_name, winner in targets:
            min_f1 = winner[min_f1_col] if min_f1_col and min_f1_col in winner else np.nan
            
            # Escape underscores for LaTeX compatibility
            res_list.append({
                'Dataset': ds.replace('_', ' '),
                'Target': target_name,
                'Model': winner['ftr_extractor'].replace('_', '\\_'),
                'CL': winner['classifier'],
                'd': int(winner['d']),
                'k': float(winner['k']),
                'OA (%)': winner['tst_oa'] * 100,
                'AA (%)': winner['tst_aa'] * 100,
                'F1_Macro (%)': winner['tst_f1_macro'] * 100,
                'Min Class': min_class.replace('_', '\\_') if min_class != "N/A" else "N/A",
                'F1_Min (%)': min_f1 * 100
            })
            
    df_table = pd.DataFrame(res_list)
    
    # 1. Terminal Report
    print("\n--- METRIC DIVERGENCE REPORT ---")
    print(tabulate(df_table, headers='keys', tablefmt='github', floatfmt=".2f"))
    
    # 2. LaTeX Export (.tex)
    latex_path = os.path.join(output_dir, 'phase1_metric_divergence_table.tex')
    try:
        latex_str = df_table.style.format("{:.2f}", subset=['k', 'OA (%)', 'AA (%)', 'F1_Macro (%)', 'F1_Min (%)']) \
                                  .hide(axis="index") \
                                  .to_latex(column_format="llccrrccccc", hrules=True)
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f">>> [SUCCESS] LaTeX code exported: {latex_path}")
    except Exception as e:
        print(f">>> [ERROR] Failed to generate LaTeX file: {e}")

    # 3. Export Image
    output_path = os.path.join(output_dir, 'phase1_metric_divergence_table.png')
    if HAS_DFI:
        try:
            df_styled = df_table.style.format("{:.2f}", subset=['k', 'OA (%)', 'AA (%)', 'F1_Macro (%)', 'F1_Min (%)']) \
                                      .background_gradient(cmap='Blues', subset=['OA (%)', 'AA (%)', 'F1_Macro (%)', 'F1_Min (%)'])
            dfi.export(df_styled, output_path, dpi=300)
            print(f">>> [SUCCESS] Table image exported: {output_path}")
        except Exception as e:
            print(f">>> [WARNING] dfi.export failed: {e}. Using matplotlib fallback.")
            HAS_DFI_INTERNAL = False
        else:
            HAS_DFI_INTERNAL = True
    else:
        HAS_DFI_INTERNAL = False

    if not HAS_DFI_INTERNAL:
        fig, ax = plt.subplots(figsize=(14, len(df_table)*0.5 + 1))
        ax.axis('off')
        tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.5)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f">>> [SUCCESS] Table image exported (fallback): {output_path}")

        
# ==========================================
# PHASE 2: CLASSIFIER BOXPLOT (ETIQUETAS Y ORDEN)
# ==========================================
def generate_phase2_boxplot(df, output_dir):
    print("\n>>> [PHASE 2] Generating Classifier Stability Boxplots...")
    apply_style()
    metric_col = METRIC_MAP.get(TARGET_METRIC, 'tst_aa')
    
    # Normalización Min-Max (Vectorizada para mayor velocidad)
    df_norm = df.copy()
    norm_col = f'{metric_col}_normalized'
    
    mins = df_norm.groupby('dataset')[metric_col].transform('min')
    maxs = df_norm.groupby('dataset')[metric_col].transform('max')
    diff = maxs - mins
    df_norm[norm_col] = np.where(diff > 0, (df_norm[metric_col] - mins) / diff, 0)
    
    # 1. Calcular medianas y definir el orden descendente
    medians = df_norm.groupby('classifier')[norm_col].median().sort_values(ascending=False)
    order = medians.index.tolist()
    
    plt.figure(figsize=(10, 6))
    
    # 2. Generar el Boxplot con el orden definido y mostrando las medias
    ax = sns.boxplot(
        data=df_norm, 
        x='classifier', 
        y=norm_col,
        order=order,           # ORDENAMIENTO JERÁRQUICO
        linewidth=1.2, 
        showfliers=False,
        showmeans=True,        # MOSTRAR EL DIAMANTE DE LA MEDIA
        meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black"},
        palette=CL_COLOR_MAP
    )
    
    # 3. Añadir el valor numérico (etiqueta de datos) sobre la mediana
    for xtick, classifier in enumerate(order):
        median_val = medians[classifier]
        ax.text(
            xtick, 
            median_val + 0.015, # Ligero desplazamiento hacia arriba para no pisar la línea
            f'{median_val:.3f}', 
            horizontalalignment='center', 
            size='medium', 
            color='black', 
            weight='bold'
        )
    
    # plt.title(f'Overall Classifier Stability Across All Datasets\nBase Metric: {TARGET_METRIC}', fontweight='bold') # Title moved to caption
    plt.ylabel('F1-macro Score', fontweight='bold')
    plt.xlabel('') # Label removed as requested
    
    output_path = os.path.join(output_dir, 'phase2_classifier_selection_boxplot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> [SUCCESS] Boxplot saved: {output_path}")
# ==========================================
# PHASE 3: LATENT SPACE EVALUATION (4x2)
# ==========================================
def generate_phase3_grid(df, output_dir):
    print("\n>>> [PHASE 3] Generating Latent Convergence Grid...")
    apply_style()
    metric_col = METRIC_MAP.get(TARGET_METRIC, 'tst_aa')
    
    # Filter by target classifier
    df_sub = df[df['classifier'] == TARGET_CL].copy()
    
    datasets = [d for d in DATASETS_ORDER if d in df_sub['dataset'].unique()]
    if not datasets:
        datasets = df_sub['dataset'].unique()
        
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(K_VALUES), 
                             figsize=(16, 4 * len(datasets)), sharex=True, sharey=False)
    
    # Handle single dataset case
    if len(datasets) == 1: 
        axes = np.expand_dims(axes, axis=0)
    
    for i, ds in enumerate(datasets):
        for j, k_val in enumerate(K_VALUES):
            ax = axes[i, j]
            subset = df_sub[(df_sub['dataset'] == ds) & (df_sub['k'] == k_val)]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center')
                continue
            
            # CORRECCIÓN 1: Desactivar sombras de varianza (errorbar=None) y aumentar grosor
            sns.lineplot(
                data=subset, x='d', y=metric_col, hue='ftr_extractor',
                marker='o', markersize=6, palette=COLOR_MAP, ax=ax, 
                errorbar=None, linewidth=2.0
            )
            
            # CORRECCIÓN 2: Escala logarítmica base 2 para visualización asintótica
            ax.set_xscale('log', base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            
            # Subplot title in alphabetical notation
            subplot_letter = chr(97 + i * len(K_VALUES) + j) # 97 is 'a'
            ax.set_title(f'({subplot_letter}) {ds.replace("_", " ")} ($k={k_val}$)', fontsize=12)
            ax.set_ylabel(f'${TARGET_METRIC}$' if j == 0 else "")
            ax.set_xlabel("Latent Dimensions ($d$)" if i == len(datasets)-1 else "")
            
            if ax.get_legend():
                ax.get_legend().remove()
            
    # Single Legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               ncol=6, bbox_to_anchor=(0.5, -0.05)) # Removed title, set ncol=6
    
    # plt.suptitle(f'Performance Convergence (${TARGET_METRIC}$) - Classifier: ${TARGET_CL}$', y=1.02) # Moved to LaTeX caption
    plt.tight_layout()
    # Add extra space at the bottom for the legend
    plt.subplots_adjust(bottom=0.08)
    
    output_path = os.path.join(output_dir, 'phase3_latent_convergence_grid.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> [SUCCESS] Convergence grid saved: {output_path}")


# ==========================================
# PHASE 4: GRANULAR CLASS-WISE ANALYSIS (OPTION B: HEATMAP)
# ==========================================
def generate_phase4_heatmap(df, output_dir):
    print("\n>>> [PHASE 4 - Option B] Generating Granular Class-wise Heatmaps...")
    apply_style()
    
    # Strict filters
    df_sub = df[
        (df['classifier'] == TARGET_CL) & 
        (df['k'] == TARGET_K) & 
        (df['d'] == TARGET_D)
    ].copy()
    
    datasets = df_sub['dataset'].unique()
    
    for ds in datasets:
        ds_data = df_sub[df_sub['dataset'] == ds]
        class_stats = DATASET_REGISTRY.get(ds, {}).get("class_stats", {})
        
        if not class_stats:
            continue
            
        f1_cols = [f'tst_f1_c{i}' for i in range(len(class_stats))]
        
        # 1. Group and average k-folds to get a single value per model/class
        df_mean = ds_data.groupby('ftr_extractor')[f1_cols].mean().reset_index()
        
        # 2. Melt to map classes
        df_melt = df_mean.melt(
            id_vars=['ftr_extractor'], value_vars=f1_cols, 
            var_name='Class_Idx', value_name='F1_Score'
        )
        
        # Map class names, N and Percentage
        class_names = list(class_stats.keys())
        class_counts = list(class_stats.values())
        total_samples = sum(class_counts)
        
        def map_class(idx_str):
            match = re.search(r'c(\d+)', idx_str)
            if not match: return idx_str
            idx = int(match.group(1))
            name = class_names[idx]
            n = class_counts[idx]
            pct = (n / total_samples) * 100
            # Use non-breaking space after comma to keep parentheses content together
            full_label = f"{name} ({pct:.1f}%,\u00A0n={n})"
            return textwrap.fill(full_label, width=22)
            
        df_melt['Class_Label'] = df_melt['Class_Idx'].apply(map_class)
        
        # 3. Pivot to create the Heatmap matrix (Classes vs Models)
        df_pivot = df_melt.pivot(index='Class_Label', columns='ftr_extractor', values='F1_Score')
        
        # Sort classes from most frequent to least frequent
        sorted_class_names = sorted(class_stats.keys(), key=lambda x: class_stats[x], reverse=True)
        df_pivot = df_pivot.reindex(sorted_class_names)
        
        # 4. Plot Heatmap
        # Adjust figure dimensions dynamically
        fig_width = max(12, len(class_stats) * 0.8) 
        fig_height = max(6, len(class_stats) * 0.52) 
        plt.figure(figsize=(fig_width, fig_height))
        
        # Configure colorbar to match axes label sizes (around 12)
        cbar_options = {'label': '$F_1$-Score'}
        
        ax = sns.heatmap(
            df_pivot, 
            annot=True,         # Show numeric value
            fmt=".2f",          # 2 decimals
            cmap="Purples",        # Premium academic color scale
            vmin=0.0, vmax=1.0, # Scale 0 to 1
            linewidths=0.5,
            cbar_kws=cbar_options
        )
        
        # Access the colorbar to change its label font size
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(12)
        cbar.ax.tick_params(labelsize=12)
        
        # plt.title(...) moved to LaTeX caption
        plt.xticks(rotation=45, ha='right', fontsize=12) 
        plt.yticks(rotation=0, fontweight='bold', fontsize=10, ha='center') 
        plt.tick_params(axis='y', pad=50) 
        plt.ylabel('') 
        plt.xlabel('') 
        
        plt.subplots_adjust(left=0.3) # Give more space to the class labels
        
        output_path = os.path.join(output_dir, f'phase4_granular_analysis_{ds}_heatmap.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f">>> [SUCCESS] Granular heatmap saved: {output_path}")

# ==========================================
# PHASE 4 (OPTION B): DELTA HEATMAP (k=1.0 vs k=0.25)
# ==========================================
def generate_phase4_delta_heatmap(df, output_dir):
    print("\n>>> [PHASE 4 - Option B] Generating Delta (k=1.0 - k=0.25) Heatmaps...")
    apply_style()
    ensure_directory(output_dir)
    
    # Strict filters: we need data for TARGET_CL and TARGET_D, but both k=1.0 and k=0.25
    df_sub = df[
        (df['classifier'] == TARGET_CL) & 
        (df['d'] == TARGET_D) &
        (df['k'].isin([1.0, 0.25]))
    ].copy()
    
    datasets = df_sub['dataset'].unique()
    
    for ds in datasets:
        ds_data = df_sub[df_sub['dataset'] == ds]
        class_stats = DATASET_REGISTRY.get(ds, {}).get("class_stats", {})
        
        if not class_stats:
            continue
            
        f1_cols = [f'tst_f1_c{i}' for i in range(len(class_stats))]
        
        # 1. Group by model and k, average folds
        df_mean = ds_data.groupby(['ftr_extractor', 'k'])[f1_cols].mean().reset_index()
        
        # 2. Separate k=1.0 and k=0.25
        df_100 = df_mean[df_mean['k'] == 1.0].set_index('ftr_extractor')[f1_cols]
        df_025 = df_mean[df_mean['k'] == 0.25].set_index('ftr_extractor')[f1_cols]
        
        # If any model is missing in either, it will be NaN. We drop those or fill.
        df_delta = df_100.subtract(df_025).dropna()
        
        if df_delta.empty:
            continue
            
        # 3. Rename columns to map class names and add percentages
        class_names = list(class_stats.keys())
        class_counts = list(class_stats.values())
        total_samples = sum(class_counts)
        
        rename_dict = {}
        for i in range(len(class_names)):
            name = class_names[i]
            n = class_counts[i]
            pct = (n / total_samples) * 100
            # Use non-breaking space after comma to keep parentheses content together
            full_label = f"{name} ({pct:.1f}%,\u00A0n={n})"
            rename_dict[f'tst_f1_c{i}'] = textwrap.fill(full_label, width=22)
            
        df_delta = df_delta.rename(columns=rename_dict)
        
        # Transpose to have Classes on Y and Models on X
        df_delta = df_delta.T
        
        # Sort classes from most frequent to least frequent
        sorted_indices = sorted(range(len(class_names)), key=lambda i: class_counts[i], reverse=True)
        sorted_labels = [rename_dict[f'tst_f1_c{i}'] for i in sorted_indices]
        df_delta = df_delta.reindex(sorted_labels)
        
        # 4. Plot Heatmap
        fig_width = max(12, len(class_stats) * 0.8) 
        fig_height = max(6, len(class_stats) * 0.52) 
        plt.figure(figsize=(fig_width, fig_height))
        
        cbar_options = {'label': '$\Delta F_1$-Score ($k=1.0 - k=0.25$)'}
        
        # Custom Diverging colormap: Orange (Negative) -> White -> Green (Positive)
        cmap_orange_green = LinearSegmentedColormap.from_list("OrangeWhiteGreen", ["#e67e22", "#ffffff", "#2ecc71"])
        
        # Find max absolute value to make the colormap symmetric
        max_abs = max(abs(df_delta.min().min()), abs(df_delta.max().max()))
        if np.isnan(max_abs) or max_abs == 0:
            max_abs = 0.1 # default safe value
            
        ax = sns.heatmap(
            df_delta, 
            annot=True,         
            fmt=".2f",          
            cmap=cmap_orange_green, 
            center=0.0,
            vmin=-max_abs, vmax=max_abs,
            linewidths=0.5,
            cbar_kws=cbar_options
        )
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(12)
        cbar.ax.tick_params(labelsize=12)
        
        plt.xticks(rotation=45, ha='right', fontsize=12) 
        plt.yticks(rotation=0, fontweight='bold', fontsize=10, ha='center') 
        plt.tick_params(axis='y', pad=50) 
        plt.ylabel('') 
        plt.xlabel('') 
        
        plt.subplots_adjust(left=0.3)
        
        output_path = os.path.join(output_dir, f'phase4_delta_analysis_{ds}_heatmap.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f">>> [SUCCESS] Delta heatmap saved: {output_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="JD Technical Report Visualization Tool")
    parser.add_argument("input", nargs='?', default="Reports", help="Input folder containing CSVs")
    parser.add_argument("output", nargs='?', default="ResultsF", help="Output folder for images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f">>> [ERROR] Input folder does not exist: {args.input}")
        return

    # Load all combined performance CSVs
    all_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith("_Combined_Performance.csv")]
    
    if not all_files:
        print(">>> [ERROR] No *_Combined_Performance.csv files found.")
        return
    
    print(f">>> Loading {len(all_files)} data files...")
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        df_list.append(temp_df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Normalize dataset names (replace spaces with underscores)
    df_all['dataset'] = df_all['dataset'].str.replace(' ', '_')
    
    ensure_directory(args.output)
    
    # Run Analysis Phases
    generate_phase1_table(df_all, args.output)
    generate_phase2_boxplot(df_all, args.output)
    generate_phase3_grid(df_all, args.output)
    generate_phase4_heatmap(df_all, args.output)  # Granular Heatmap
    
    # Run Delta Heatmap in ResultsF2
    output_dir2 = "ResultsF2"
    generate_phase4_delta_heatmap(df_all, output_dir2)
    
    print(f"\n>>> [DONE] Main reports generated in: {args.output}")
    print(f">>> [DONE] Delta reports generated in: {output_dir2}")

if __name__ == "__main__":
    main()
