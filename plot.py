import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import argparse
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def load_and_prepare_data(csv_file):
    """Load CSV data and extract L1 and L2 configuration values"""
    df = pd.read_csv(csv_file)
    
    # Extract L1 config value from test name
    df['L1_Config'] = df['Test Name'].apply(
        lambda x: int(re.search(r'l1config-(\d+)', str(x)).group(1)) 
        if re.search(r'l1config-(\d+)', str(x)) else np.nan
    )
    
    # Extract L2 config value from test name
    df['L2_Config'] = df['Test Name'].apply(
        lambda x: int(re.search(r'l2config-(\d+)', str(x)).group(1)) 
        if re.search(r'l2config-(\d+)', str(x)) else np.nan
    )
    
    # Convert to % of shared memory (L1 config is reversed)
    df['L1_Config'] = 100 - df['L1_Config']
    
    return df.dropna(subset=['L1_Config', 'L2_Config'])

def filter_flash_attn_kernels(df, min_execution_time=5000):
    """Filter relevant Flash Attention computation kernels"""
    # Filter out very small kernels (likely initialization)
    df = df[df['Kernel execution time (ns)'] > min_execution_time]
    return df

def create_metric_plots(df, output_dir, metric_name, display_name=None, config_name="", plot_type='heatmap'):
    """Create plots for a single metric, showing its relationship with L1 and L2 configurations"""
    os.makedirs(output_dir, exist_ok=True)
    
    if display_name is None:
        display_name = metric_name
    
    title_prefix = f"{config_name} - " if config_name else ""
    
    # Create a pivot table with L1_Config as rows, L2_Config as columns, and the metric as values
    pivot_df = df.groupby(['L1_Config', 'L2_Config'])[metric_name].mean().reset_index()
    pivot_table = pivot_df.pivot(index='L1_Config', columns='L2_Config', values=metric_name)
    
    # Heatmap Plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', 
                    cbar_kws={'label': display_name})
    plt.title(f'{title_prefix}{display_name} vs L1/L2 Configuration', fontsize=14)
    plt.xlabel('L2 Configuration (% persisting cache)', fontsize=12)
    plt.ylabel('L1 Configuration (% shared memory)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric_name}_heatmap.png', dpi=300)
    plt.close()
    
    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    l1_configs = pivot_table.index.values
    l2_configs = pivot_table.columns.values
    X, Y = np.meshgrid(l2_configs, l1_configs)
    Z = pivot_table.values
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('L2 Configuration (% persisting cache)')
    ax.set_ylabel('L1 Configuration (% shared memory)')
    ax.set_zlabel(display_name)
    ax.set_title(f'{title_prefix}{display_name} vs L1/L2 Configuration')
    
    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5, label=display_name)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric_name}_3d.png', dpi=300)
    plt.close()
    
    # Line plots - show L2 impact for different L1 settings
    plt.figure(figsize=(12, 8))
    for l1 in pivot_table.index.unique():
        l1_data = pivot_df[pivot_df['L1_Config'] == l1]
        plt.plot(l1_data['L2_Config'], l1_data[metric_name], 'o-', 
                linewidth=2, label=f'L1 = {l1}%')
    
    plt.title(f'{title_prefix}{display_name} vs L2 Configuration (For Different L1 Settings)', fontsize=14)
    plt.xlabel('L2 Configuration (% persisting cache)', fontsize=12)
    plt.ylabel(display_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric_name}_by_L1.png', dpi=300)
    plt.close()
    
    # Line plots - show L1 impact for different L2 settings
    plt.figure(figsize=(12, 8))
    for l2 in pivot_table.columns.unique():
        l2_data = pivot_df[pivot_df['L2_Config'] == l2]
        plt.plot(l2_data['L1_Config'], l2_data[metric_name], 's-', 
                linewidth=2, label=f'L2 = {l2}%')
    
    plt.title(f'{title_prefix}{display_name} vs L1 Configuration (For Different L2 Settings)', fontsize=14)
    plt.xlabel('L1 Configuration (% shared memory)', fontsize=12)
    plt.ylabel(display_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric_name}_by_L2.png', dpi=300)
    plt.close()

def process_subdirectory(subdir_path):
    """Process metrics CSV in a subdirectory and generate plots for each metric"""
    config_name = os.path.basename(subdir_path)
    print(f"Processing plots for configuration: {config_name}")
    
    # Check if metrics file exists
    csv_file = os.path.join(subdir_path, "ncu_metrics.csv")
    if not os.path.exists(csv_file):
        print(f"  No metrics file found at {csv_file}")
        return False
    
    # Create plots directory inside the subdirectory
    plots_dir = os.path.join(subdir_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(csv_file)
        if df.empty:
            print(f"  No valid L1/L2 configuration data found in {csv_file}")
            return False
        
        # Filter for meaningful kernels
        flash_attn_df = filter_flash_attn_kernels(df)
        if flash_attn_df.empty:
            print(f"  No kernels with significant execution time found in {csv_file}")
            return False
        
        # Define metrics to plot (each will get its own set of plots)
        metrics_to_plot = [
            # Cache metrics
            ('Unified L1 cache hit rate', 'L1 Cache Hit Rate (%)'),
            ('Unified L1 cache hit rate for read transactions (global memory accesses)', 'L1 Global Memory Read Hit Rate (%)'),
            ('L2 cache hit rate', 'L2 Cache Hit Rate (%)'),
            
            # Performance metrics
            ('Kernel execution time (ns)', 'Execution Time (ns)'),
            ('Instructions executed per clock cycle (IPC)', 'IPC'),
            
            # Occupancy metrics
            ('Achieved occupancy', 'Achieved Occupancy (%)'),
            ('Achieved active warps per SM', 'Active Warps per SM'),
            ('Theoretical occupancy', 'Theoretical Occupancy (%)'),
            
            # Memory transaction metrics
            ('GMEM total transactions', 'Global Memory Transactions'),
            ('L2 total transactions', 'L2 Cache Transactions'),
            ('DRAM total transactions', 'DRAM Transactions'),
            
            # Additional metrics
            ('Thread block limit shared memory', 'Thread Block Limit (Shared Memory)'),
            ('Thread block limit registers', 'Thread Block Limit (Registers)'),
            ('Unified L1 cache total requests', 'L1 Cache Total Requests'),
            ('Unified L2 cache total requests', 'L2 Cache Total Requests')
        ]
        
        # Generate individual plots for each metric
        for metric_name, display_name in metrics_to_plot:
            if metric_name in flash_attn_df.columns:
                print(f"  Generating plots for metric: {metric_name}")
                create_metric_plots(flash_attn_df, plots_dir, metric_name, display_name, config_name)
            else:
                print(f"  Metric {metric_name} not found in data")
        
        # For the top kernels, create separate plots
        kernel_execution_times = flash_attn_df.groupby('Kernel ID')['Kernel execution time (ns)'].mean()
        top_kernels = kernel_execution_times.nlargest(3).index.tolist()
        
        for kid in top_kernels:
            kernel_df = flash_attn_df[flash_attn_df['Kernel ID'] == kid]
            kernel_output_dir = os.path.join(plots_dir, f'kernel_{kid}')
            os.makedirs(kernel_output_dir, exist_ok=True)
            
            # Generate plots for each metric for this specific kernel
            for metric_name, display_name in metrics_to_plot:
                if metric_name in kernel_df.columns:
                    create_metric_plots(kernel_df, kernel_output_dir, metric_name, 
                                       display_name, f"{config_name} - Kernel {kid}")
        
        print(f"  Successfully generated plots for {config_name}")
        return True
    
    except Exception as e:
        print(f"  Error processing {csv_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_comparative_plots(root_dir):
    """Create comparative plots across all configurations"""
    # Find all subdirectories with metrics files
    all_metrics_files = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            csv_file = os.path.join(subdir_path, "ncu_metrics.csv")
            if os.path.exists(csv_file):
                all_metrics_files.append(csv_file)
    
    if not all_metrics_files:
        print("No metrics files found for comparative analysis")
        return
    
    print(f"Creating comparative plots across {len(all_metrics_files)} configurations")
    
    # Create a directory for comparative plots
    comparative_dir = os.path.join(root_dir, "comparative_plots")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Load and combine all data
    all_data = []
    for csv_file in all_metrics_files:
        df = load_and_prepare_data(csv_file)
        if not df.empty:
            config_name = os.path.basename(os.path.dirname(csv_file))
            df['Configuration'] = config_name
            all_data.append(df)
    
    if not all_data:
        print("No valid data found for comparative analysis")
        return
    
    combined_df = pd.concat(all_data)
    filtered_df = filter_flash_attn_kernels(combined_df)
    
    if filtered_df.empty:
        print("No significant kernels found for comparative analysis")
        return
    
    # Define key metrics for comparison
    key_metrics = [
        'Kernel execution time (ns)',
        'L2 cache hit rate',
        'Unified L1 cache hit rate',
        'Instructions executed per clock cycle (IPC)',
        'Achieved occupancy'
    ]
    
    # For each key metric, create a plot showing how it varies across configurations
    for metric in key_metrics:
        if metric in filtered_df.columns:
            plt.figure(figsize=(15, 10))
            
            # Calculate mean value for each configuration
            agg_df = filtered_df.groupby(['Configuration', 'L1_Config', 'L2_Config'])[metric].mean().reset_index()
            
            # Sort by performance (lower execution time is better, higher hit rates are better)
            sort_ascending = True if metric == 'Kernel execution time (ns)' else False
            agg_df = agg_df.sort_values(by=metric, ascending=sort_ascending)
            
            # Plot the top 20 configurations
            top_df = agg_df.head(20)
            
            # Create configuration labels combining L1 and L2 values
            top_df['Config_Label'] = top_df.apply(
                lambda x: f"L1:{x['L1_Config']}%,L2:{x['L2_Config']}%", axis=1)
            
            sns.barplot(x='Config_Label', y=metric, data=top_df)
            plt.xticks(rotation=90)
            plt.title(f'Top 20 Configurations for {metric}', fontsize=14)
            plt.xlabel('Configuration (L1%, L2%)', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(comparative_dir, f'{metric.replace(" ", "_")}_comparison.png'), dpi=300)
            plt.close()
    
    print("Comparative plots created successfully")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate plots from NCU metrics')
    parser.add_argument('--root-dir', type=str, default='./ncu_reports', 
                      help='Root directory containing subdirectories with NCU metrics')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    root_dir = args.root_dir
    
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Find all subdirectories in the root directory
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {root_dir}")
        return
    
    print(f"Found {len(subdirs)} configuration directories to process")
    
    # Process each subdirectory
    success_count = 0
    for subdir in subdirs:
        if process_subdirectory(subdir):
            success_count += 1
    
    print(f"Analysis complete. Successfully processed {success_count} out of {len(subdirs)} configurations.")
    
    # Generate comparative plots across configurations
    create_comparative_plots(root_dir)
    
if __name__ == "__main__":
    main()
