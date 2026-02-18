#!/usr/bin/env python3
"""
Plot weight distributions (lora_B @ lora_A) for all adapters from a training run.
Creates a grid of subplots (3x5 or 5x3), one for each task.

Usage:
    python tests/plot_weight_distributions.py
    python tests/plot_weight_distributions.py --output_dir logs_and_outputs/ella/long/order_5/outputs1
"""

import sys
import os
import argparse
import glob
import re
import json

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt


def natural_sort_key(s):
    """Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]


def load_adapter_weights(adapter_path):
    """Load adapter weights directly from adapter_model.bin."""
    weights_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(weights_path):
        return torch.load(weights_path, map_location='cpu')
    
    # Try safetensors format
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def compute_w_past_from_weights(state_dict):
    """
    Compute W_past = lora_B @ lora_A for all LoRA layers from state dict.
    Returns flattened array of all W_past values.
    """
    # Group weights by layer
    lora_a_weights = {}
    lora_b_weights = {}
    
    for key, value in state_dict.items():
        if 'lora_A' in key and 'loranew' not in key:
            # Extract layer identifier (everything before lora_A)
            layer_id = key.split('lora_A')[0]
            lora_a_weights[layer_id] = value
        elif 'lora_B' in key and 'loranew' not in key:
            layer_id = key.split('lora_B')[0]
            lora_b_weights[layer_id] = value
    
    # Compute W_past for each layer
    all_values = []
    for layer_id in lora_a_weights:
        if layer_id in lora_b_weights:
            lora_A = lora_a_weights[layer_id]  # [r, d_in]
            lora_B = lora_b_weights[layer_id]  # [d_out, r]
            
            # W_past = lora_B @ lora_A
            W_past = torch.mm(lora_B, lora_A)
            all_values.append(W_past.flatten().numpy())
    
    return np.concatenate(all_values) if all_values else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Plot weight distributions for all adapters")
    parser.add_argument("--output_dir", type=str, 
                        default="logs_and_outputs/ella/long/order_5/outputs1",
                        help="Directory containing task outputs with adapters (relative to project root)")
    parser.add_argument("--save_path", type=str, default="/root/projects/O-LoRA/tests/weight_distributions.png",
                        help="Path to save the plot (if not specified, shows interactively)")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths from project root
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    # Find all adapter directories
    adapter_dirs = sorted(glob.glob(os.path.join(output_dir, "*/adapter")), 
                          key=natural_sort_key)
    
    if not adapter_dirs:
        print(f"No adapters found in {output_dir}")
        return
    
    num_tasks = len(adapter_dirs)
    print(f"Found {num_tasks} adapters in {output_dir}")
    
    # Collect distributions for each adapter
    task_names = []
    raw_distributions = []
    
    for adapter_path in adapter_dirs:
        # Extract task name from path
        task_name = os.path.basename(os.path.dirname(adapter_path))
        task_names.append(task_name)
        print(f"  Loading {task_name}...")
        
        # Load adapter weights directly
        state_dict = load_adapter_weights(adapter_path)
        
        # Compute W_past values
        values = compute_w_past_from_weights(state_dict)
        raw_distributions.append(values)
        
        # Print some info
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                r_sum = config.get('r_sum', config.get('r', '?'))
                print(f"    r_sum={r_sum}, values shape: {values.shape}")
    
    # Determine global x-axis limits for consistency
    all_values = np.concatenate(raw_distributions)
    x_min = np.percentile(all_values, 0.1)
    x_max = np.percentile(all_values, 99.9)
    abs_max = np.percentile(np.abs(all_values), 99.9)
    
    # =========================================================================
    # Plot 1: Raw distributions (3x5 grid)
    # =========================================================================
    nrows, ncols = 3, 5
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(20, 12))
    axes1 = axes1.flatten()
    
    for i, (name, values) in enumerate(zip(task_names, raw_distributions)):
        ax = axes1[i]
        ax.hist(values, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add stats
        mean_val = np.mean(values)
        std_val = np.std(values)
        abs_vals = np.abs(values)
        p95 = np.percentile(abs_vals, 95)  # high-energy tail threshold
        ax.text(0.95, 0.95, f'μ={mean_val:.4f}\nσ={std_val:.4f}\n|W|₉₅%={p95:.4f}', 
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.axvline(p95, color='green', linestyle=':', linewidth=0.8, alpha=0.7, label='95% |W|')
        ax.axvline(-p95, color='green', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # Hide unused axes
    for i in range(num_tasks, len(axes1)):
        axes1[i].axis('off')
    
    fig1.suptitle('Raw W_past = lora_B @ lora_A distribution per task', fontsize=14, fontweight='bold')
    fig1.supxlabel('Weight Value', fontsize=12)
    fig1.supylabel('Density', fontsize=12)
    plt.tight_layout()
    
    if args.save_path:
        raw_path = args.save_path.replace('.png', '_raw.png')
        fig1.savefig(raw_path, dpi=150, bbox_inches='tight')
        print(f"Saved raw distribution plot to {raw_path}")
    
    # =========================================================================
    # Plot 2: Absolute distributions (3x5 grid)
    # =========================================================================
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(20, 12))
    axes2 = axes2.flatten()
    
    for i, (name, values) in enumerate(zip(task_names, raw_distributions)):
        ax = axes2[i]
        abs_values = np.abs(values)
        ax.hist(abs_values, bins=80, density=True, alpha=0.7, color='darkorange', edgecolor='none')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlim(0, abs_max)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add stats
        mean_val = np.mean(abs_values)
        std_val = np.std(abs_values)
        ax.text(0.95, 0.95, f'μ={mean_val:.4f}\nσ={std_val:.4f}', 
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for i in range(num_tasks, len(axes2)):
        axes2[i].axis('off')
    
    fig2.suptitle('Absolute |W_past| = |lora_B @ lora_A| distribution per task', fontsize=14, fontweight='bold')
    fig2.supxlabel('|Weight Value|', fontsize=12)
    fig2.supylabel('Density', fontsize=12)
    plt.tight_layout()
    
    if args.save_path:
        abs_path = args.save_path.replace('.png', '_abs.png')
        fig2.savefig(abs_path, dpi=150, bbox_inches='tight')
        print(f"Saved absolute distribution plot to {abs_path}")
    
    # =========================================================================
    # Plot 3: Log-scale absolute distributions (3x5 grid)
    # =========================================================================
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(20, 12))
    axes3 = axes3.flatten()
    
    for i, (name, values) in enumerate(zip(task_names, raw_distributions)):
        ax = axes3[i]
        abs_values = np.abs(values)
        # Filter out zeros for log scale
        nonzero_vals = abs_values[abs_values > 0]
        ax.hist(nonzero_vals, bins=80, density=True, alpha=0.7, color='seagreen', edgecolor='none')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=8)
        
        # Add stats
        mean_val = np.mean(abs_values)
        median_val = np.median(abs_values)
        ax.text(0.95, 0.95, f'μ={mean_val:.4f}\nmed={median_val:.4f}', 
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused axes
    for i in range(num_tasks, len(axes3)):
        axes3[i].axis('off')
    
    fig3.suptitle('Log-scale |W_past| = |lora_B @ lora_A| distribution per task', fontsize=14, fontweight='bold')
    fig3.supxlabel('|Weight Value| (log scale)', fontsize=12)
    fig3.supylabel('Density', fontsize=12)
    plt.tight_layout()
    
    if args.save_path:
        log_path = args.save_path.replace('.png', '_log.png')
        fig3.savefig(log_path, dpi=150, bbox_inches='tight')
        print(f"Saved log-scale distribution plot to {log_path}")
    
    if not args.save_path:
        plt.show()
    
    # Print statistics table
    print("\n" + "="*90)
    print("Statistics per task:")
    print("="*90)
    print(f"{'Task':<20} {'Mean':>12} {'Std':>12} {'|Mean|':>12} {'|Std|':>12} {'Min':>12} {'Max':>12}")
    print("-"*90)
    for name, values in zip(task_names, raw_distributions):
        print(f"{name:<20} {np.mean(values):>12.6f} {np.std(values):>12.6f} "
              f"{np.mean(np.abs(values)):>12.6f} {np.std(np.abs(values)):>12.6f} "
              f"{np.min(values):>12.6f} {np.max(values):>12.6f}")
    print("="*90)


if __name__ == "__main__":
    main()
