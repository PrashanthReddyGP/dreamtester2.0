import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

def generate_parameter_heatmap(results_data, params_to_optimize, metric, batch_id, strategy_key):
    """
    Generates and saves a heatmap grid.
    - 2 params: single 2D heatmap
    - 3 params: single row of heatmaps (1D facet)
    - 4 params: a 2D grid of heatmaps (2D facet grid)
    """
    if not results_data:
        print("--- NOTE: No results data to generate a heatmap. ---")
        return

    param_names = [p['name'] for p in params_to_optimize]
    num_params = len(param_names)

    if num_params < 2:
        print("--- NOTE: Not enough parameters (<2) to generate a heatmap. ---")
        return

    # --- Data Preparation (same as before) ---
    records = []
    for combo_tuple, metric_value in results_data:
        record = dict(zip(param_names, combo_tuple))
        record[metric] = metric_value
        records.append(record)
    
    if not records:
        print("--- NOTE: No valid records to create a heatmap. ---")
        return
        
    df = pd.DataFrame(records)
    # 1. Find the true min and max values across all results for the metric
    if df.empty or metric not in df.columns:
        print("--- WARNING: Metric column not found or DataFrame is empty. Cannot generate heatmap. ---")
        return
        
    global_min = df[metric].min()
    global_max = df[metric].max()
    
    # 2. Handle edge cases where all data is on one side of zero
    if global_min >= 0: global_min = 0 # If no losses, anchor min at 0
    if global_max <= 0: global_max = 0 # If no profits, anchor max at 0
    
    # If there's no range, create a small default one to avoid errors
    if global_min == global_max:
        global_min -= 1
        global_max += 1

    # # 1. Find the true min and max values from the data
    # true_min = df[metric].min()
    # true_max = df[metric].max()

    # # 2. Determine the symmetric limit for the color scale
    # # This finds the value with the largest distance from zero.
    # limit = max(abs(true_min), abs(true_max))

    # # If the limit is zero (i.e., all results were 0), create a small default range
    # # to prevent a zero-width scale (e.g. vmin=0, vmax=0) which would also error.
    # if limit == 0:
    #     limit = 1 # e.g., scale will be -1 to 1

    # # 3. Create a robust, symmetric TwoSlopeNorm object.
    # # The scale will now run from -limit to +limit, centered perfectly on 0.
    # # This is guaranteed to be valid because limit is always > 0 here.
    # norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    # 3. Create the TwoSlopeNorm normalizer object
    # This is the core of the fix. It maps the colors based on the actual data range
    # on either side of the center point (0).
    norm = TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)

    print(f"--- Heatmap ASYMMETRIC color scale set for '{metric}': Min={global_min:.2f}, Center=0, Max={global_max:.2f} ---")
    
    # Assign roles based on parameter order
    x_param = param_names[0]
    y_param = param_names[1]
    col_facet_param = param_names[2] if num_params >= 3 else None
    row_facet_param = param_names[3] if num_params >= 4 else None
    
    # Get unique values for faceting and ensure they are sorted correctly
    col_values = [str(v) for v in sorted(df[col_facet_param].unique())] if col_facet_param else [None]
    row_values = [str(v) for v in sorted(df[row_facet_param].unique())] if row_facet_param else [None]
    
    # Determine grid size
    n_cols = len(col_values)
    n_rows = len(row_values)
    
    # <<< DYNAMIC FIGSIZE ADJUSTMENT >>>
    # Adjust base size based on density to keep plots readable
    base_width_per_plot = 7
    base_height_per_plot = 6
    
    # Reduce size slightly for very large grids to prevent huge images
    if n_cols > 3:
        base_width_per_plot = 6
    if n_rows > 3:
        base_height_per_plot = 5
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * base_width_per_plot, n_rows * base_height_per_plot), squeeze=False)
    
    # Iterate through the 2D grid of facets
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]
            
            # Filter data for the current subplot
            subset_df = df.copy()
            title_parts = []
            
            if row_facet_param:
                subset_df = subset_df[subset_df[row_facet_param].astype(str) == row_val]
                title_parts.append(f"{row_facet_param} = {row_val}")
            if col_facet_param:
                subset_df = subset_df[subset_df[col_facet_param].astype(str) == col_val]
                title_parts.append(f"{col_facet_param} = {col_val}")
            
            ax.set_title(" | ".join(title_parts))
            
            # Check if there's enough data to form a 2D grid for the heatmap
            if subset_df.empty or subset_df[x_param].nunique() < 2 or subset_df[y_param].nunique() < 1:
                ax.text(0.5, 0.5, 'Not enough data to form a 2D plot', ha='center', va='center')
                continue
            
            try:
                pivot_df = subset_df.pivot_table(index=y_param, columns=x_param, values=metric)
                if not pivot_df.empty:
                    sns.heatmap(
                        pivot_df, 
                        ax=ax, 
                        cmap="RdYlGn", 
                        annot=True, 
                        fmt=".2f",
                        norm=norm  # Use our new asymmetric normalizer
                    )
                    ax.invert_yaxis()
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting:\n{e}', ha='center', va='center')

    # --- Final Figure Formatting ---
    main_title = f'Optimization Heatmap for Strategy {strategy_key.upper()} ({metric})'
    if row_facet_param:
        main_title += f'\nRows: {row_facet_param} | Columns: {col_facet_param}'
    elif col_facet_param:
        main_title += f'\nFaceted by {col_facet_param}'

    fig.suptitle(main_title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plot_dir = os.path.join('params_distribution')
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"heatmap_strat_{strategy_key}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"--- Heatmap saved to {filepath} ---")