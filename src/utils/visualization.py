"""Visualization utilities for experiment results."""

import json
import os
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})


def plot_training_loss(
    history: Dict[str, Any],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss')
    
    if 'valid_loss' in history:
        ax.plot(epochs, history['valid_loss'], 'r--', lw=2, label='Valid Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Cross-Entropy)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_loglik_vs_hyperparam(
    param_values: List,
    logliks: List[float],
    param_name: str,
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot test log-likelihood vs hyperparameter values.
    
    Args:
        param_values: List of hyperparameter values
        logliks: Corresponding log-likelihood values
        param_name: Name of the hyperparameter
        model_name: Name of the model
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(param_values, logliks, 'bo-', markersize=10, lw=2)
    
    # Mark best
    best_idx = np.argmax(logliks)
    ax.plot(
        param_values[best_idx], logliks[best_idx], 'r*',
        markersize=20, label=f'Best: {param_values[best_idx]}'
    )
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Test Log-Likelihood')
    ax.set_title(f'{model_name}: Test LL vs {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_flops_trajectory(
    histories: Dict[str, Dict],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot validation log-likelihood vs cumulative training FLOPs.
    
    Args:
        histories: Dictionary mapping model names to training histories
        title: Plot title
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(histories), 10)))
    
    for (name, hist), color in zip(histories.items(), colors):
        if 'flops_per_epoch' in hist and 'valid_loglik' in hist:
            flops = hist['flops_per_epoch']
            ll = hist['valid_loglik']
            min_len = min(len(flops), len(ll))
            ax.plot(flops[:min_len], ll[:min_len], '-', color=color, lw=2, label=name)
    
    ax.set_xscale('log')
    ax.set_xlabel('Cumulative Training FLOPs')
    ax.set_ylabel('Validation Log-Likelihood')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison_bar(
    histories: Dict[str, Dict],
    metric: str = 'test_loglik',
    title: str = 'Model Comparison',
    save_path: Optional[str] = None,
) -> None:
    """
    Create horizontal bar chart comparing models on a metric.
    
    Args:
        histories: Dictionary mapping model names to training histories
        metric: Metric to compare
        title: Plot title
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(histories.keys())
    values = []
    for n in names:
        val = histories[n].get(metric)
        values.append(val if val is not None else 0)
    
    # Sort by value
    pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    names, values = zip(*pairs)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=8
        )
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_architecture_comparison(
    all_histories: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Create 4-panel comparison of architectures.
    
    Args:
        all_histories: Dictionary mapping model names to training histories
        save_path: Path to save the figure
    """
    from matplotlib.patches import Patch
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by architecture
    arch_groups = {}
    for name, hist in all_histories.items():
        parts = name.split('_')
        arch = parts[0]
        if arch not in arch_groups:
            arch_groups[arch] = {}
        arch_groups[arch][name] = hist
    
    arch_colors = {
        'linear': 'tab:blue',
        'mlp': 'tab:green',
        'attention': 'tab:orange',
        'transformer': 'tab:red',
        'ptb': 'tab:purple',
        'wikitext': 'tab:brown',
    }
    
    # Panel 1: Training curves
    ax = axes[0, 0]
    for arch, group in arch_groups.items():
        if not group:
            continue
        best_name = max(
            group.keys(),
            key=lambda x: group[x].get('test_loglik') or -999
        )
        hist = group[best_name]
        if 'train_loss' in hist:
            epochs = range(1, len(hist['train_loss']) + 1)
            ax.plot(
                epochs, hist['train_loss'], '-',
                color=arch_colors.get(arch, 'gray'), lw=2, label=arch
            )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Curves (Best per Architecture)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Best performance bar chart
    ax = axes[0, 1]
    arch_logliks = {}
    for arch, group in arch_groups.items():
        lls = [h.get('test_loglik') or -999 for h in group.values()]
        arch_logliks[arch] = max(lls) if lls else -999
    
    archs = [a for a in arch_logliks.keys() if arch_logliks[a] > -999]
    lls = [arch_logliks[a] for a in archs]
    colors = [arch_colors.get(a, 'gray') for a in archs]
    
    if archs:
        ax.bar(archs, lls, color=colors)
        ax.set_ylabel('Best Test Log-Likelihood')
        ax.set_title('Best Performance by Architecture')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Compute efficiency scatter
    ax = axes[1, 0]
    for arch, group in arch_groups.items():
        for name, hist in group.items():
            flops = hist.get('total_flops', 0)
            ll = hist.get('test_loglik')
            if flops > 0 and ll is not None:
                ax.scatter(
                    flops, ll, s=80, color=arch_colors.get(arch, 'gray'),
                    alpha=0.7, edgecolors='black', linewidth=0.5
                )
    
    legend_items = [
        Patch(facecolor=c, label=a)
        for a, c in arch_colors.items() if a in arch_groups
    ]
    if legend_items:
        ax.legend(handles=legend_items)
    ax.set_xscale('log')
    ax.set_xlabel('Training FLOPs')
    ax.set_ylabel('Test Log-Likelihood')
    ax.set_title('Compute Efficiency')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary text
    ax = axes[1, 1]
    summary = "Performance Summary\n" + "=" * 30 + "\n\n"
    for arch in ['linear', 'mlp', 'attention', 'transformer']:
        if arch in arch_logliks and arch_logliks[arch] > -999:
            summary += f"{arch.capitalize():12s}: {arch_logliks[arch]:.4f}\n"
    
    ax.text(
        0.5, 0.5, summary, ha='center', va='center',
        fontsize=12, transform=ax.transAxes, family='monospace'
    )
    ax.axis('off')
    
    plt.suptitle('Architecture Comparison - Tiny Shakespeare', fontsize=14, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def load_history(path: str) -> Dict:
    """Load training history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_all_histories(results_dir: str) -> Dict[str, Dict]:
    """
    Load all training histories from a results directory.
    
    Args:
        results_dir: Directory containing *_history.json files
        
    Returns:
        Dictionary mapping model names to training histories
    """
    histories = {}
    if not os.path.exists(results_dir):
        return histories
    
    for fname in os.listdir(results_dir):
        if fname.endswith('_history.json'):
            name = fname.replace('_history.json', '')
            try:
                histories[name] = load_history(os.path.join(results_dir, fname))
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    return histories
