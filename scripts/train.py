#!/usr/bin/env python3
"""
Main training script for language model experiments.

Examples:
    python scripts/train.py                      # Run all experiments
    python scripts/train.py --model transformer  # Train specific model
    python scripts/train.py --quick              # Quick test run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import LinearPredictor, MLPLanguageModel, AttentionLanguageModel, TransformerLM
from data import load_shakespeare, load_ptb, load_wikitext, create_dataloaders
from training import train_model, generate_text_char, generate_text_word
from utils import (
    plot_training_loss, plot_loglik_vs_hyperparam,
    plot_architecture_comparison, plot_flops_trajectory,
    plot_model_comparison_bar, load_all_histories
)

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def setup_seeds(seed: int = SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history: dict, name: str) -> str:
    """Save training history to JSON."""
    path = RESULTS_DIR / f"{name}_history.json"
    h = {}
    for k, v in history.items():
        if isinstance(v, list):
            h[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        elif hasattr(v, 'item'):
            h[k] = float(v)
        else:
            h[k] = v
    with open(path, 'w') as f:
        json.dump(h, f, indent=2)
    return str(path)


def run_linear_experiments(vocab_size: int, quick: bool = False):
    """Train linear models with varying context lengths."""
    print("\n" + "=" * 50)
    print("LINEAR PREDICTOR EXPERIMENTS")
    print("=" * 50)
    
    cfg = {
        'epochs': 5 if quick else 15,
        'lr': 0.01,
        'batch_size': 128,
        'context_lengths': [32, 64] if quick else [16, 32, 64, 128],
    }
    
    results = {}
    for ctx_len in cfg['context_lengths']:
        print(f"\n--- Context Length: {ctx_len} ---")
        
        train_ds, valid_ds, test_ds = load_shakespeare(str(DATA_DIR), ctx_len)
        loaders = create_dataloaders(train_ds, valid_ds, test_ds, cfg['batch_size'])
        
        model = LinearPredictor(vocab_size, ctx_len)
        print(f"Parameters: {model.num_parameters:,}")
        
        history = train_model(
            model, *loaders, cfg['epochs'], cfg['lr'], DEVICE,
            save_path=str(RESULTS_DIR / f"linear_ctx{ctx_len}.pt")
        )
        
        name = f"linear_ctx{ctx_len}"
        save_history(history, name)
        results[name] = history
        
        plot_training_loss(history, f"Linear (ctx={ctx_len})",
                          str(FIGURES_DIR / f"{name}_training.png"))
    
    # Comparison plot
    ctx_vals = cfg['context_lengths']
    logliks = [results[f"linear_ctx{c}"]["test_loglik"] for c in ctx_vals]
    plot_loglik_vs_hyperparam(ctx_vals, logliks, "Context Length", "Linear",
                              str(FIGURES_DIR / "linear_comparison.png"))
    
    best_ctx = ctx_vals[np.argmax(logliks)]
    print(f"\nBest Linear: ctx={best_ctx}, loglik={max(logliks):.4f}")
    return results


def run_mlp_experiments(vocab_size: int, quick: bool = False):
    """Train MLP models with varying hidden dimensions."""
    print("\n" + "=" * 50)
    print("MLP EXPERIMENTS")
    print("=" * 50)
    
    cfg = {
        'epochs': 8 if quick else 25,
        'lr': 0.001,
        'batch_size': 128,
        'context_length': 64,
        'embed_dim': 32,
        'hidden_dims_list': [[256]*3] if quick else [[128]*3, [256]*3, [512]*3],
    }
    
    train_ds, valid_ds, test_ds = load_shakespeare(str(DATA_DIR), cfg['context_length'])
    loaders = create_dataloaders(train_ds, valid_ds, test_ds, cfg['batch_size'])
    
    results = {}
    for hdims in cfg['hidden_dims_list']:
        hdim_str = "x".join(map(str, hdims))
        print(f"\n--- Hidden: {hdims} ---")
        
        model = MLPLanguageModel(vocab_size, cfg['context_length'],
                                 embed_dim=cfg['embed_dim'], hidden_dims=hdims)
        print(f"Parameters: {model.num_parameters:,}")
        
        history = train_model(
            model, *loaders, cfg['epochs'], cfg['lr'], DEVICE,
            save_path=str(RESULTS_DIR / f"mlp_{hdim_str}.pt")
        )
        
        name = f"mlp_{hdim_str}"
        save_history(history, name)
        results[name] = history
        
        plot_training_loss(history, f"MLP ({hdim_str})",
                          str(FIGURES_DIR / f"{name}_training.png"))
    
    print(f"\nMLP experiments complete")
    return results


def run_attention_experiments(vocab_size: int, quick: bool = False):
    """Train attention models with varying number of heads."""
    print("\n" + "=" * 50)
    print("SELF-ATTENTION EXPERIMENTS")
    print("=" * 50)
    
    cfg = {
        'epochs': 10 if quick else 30,
        'lr': 0.001,
        'batch_size': 64,
        'context_length': 64,
        'embed_dim': 128,
        'n_layers': 2,
        'heads_list': [2, 4] if quick else [1, 2, 4, 8],
    }
    
    train_ds, valid_ds, test_ds = load_shakespeare(str(DATA_DIR), cfg['context_length'])
    loaders = create_dataloaders(train_ds, valid_ds, test_ds, cfg['batch_size'])
    
    results = {}
    for n_heads in cfg['heads_list']:
        print(f"\n--- Heads: {n_heads} ---")
        
        model = AttentionLanguageModel(
            vocab_size, cfg['context_length'],
            embed_dim=cfg['embed_dim'], n_heads=n_heads, n_layers=cfg['n_layers']
        )
        print(f"Parameters: {model.num_parameters:,}")
        
        history = train_model(
            model, *loaders, cfg['epochs'], cfg['lr'], DEVICE,
            save_path=str(RESULTS_DIR / f"attention_h{n_heads}.pt")
        )
        
        name = f"attention_h{n_heads}"
        save_history(history, name)
        results[name] = history
        
        plot_training_loss(history, f"Attention ({n_heads} heads)",
                          str(FIGURES_DIR / f"{name}_training.png"))
    
    print(f"\nAttention experiments complete")
    return results


def run_transformer_experiments(vocab_size: int, quick: bool = False):
    """Train transformer models with varying depths."""
    print("\n" + "=" * 50)
    print("TRANSFORMER EXPERIMENTS")
    print("=" * 50)
    
    cfg = {
        'epochs': 12 if quick else 40,
        'lr': 3e-4,
        'batch_size': 32,
        'context_length': 128,
        'embed_dim': 128,
        'n_heads': 4,
        'layers_list': [2, 4] if quick else [2, 4, 6],
    }
    
    train_ds, valid_ds, test_ds = load_shakespeare(
        str(DATA_DIR), cfg['context_length'], for_transformer=True
    )
    loaders = create_dataloaders(train_ds, valid_ds, test_ds, cfg['batch_size'])
    
    results = {}
    for n_layers in cfg['layers_list']:
        print(f"\n--- Layers: {n_layers} ---")
        
        model = TransformerLM(
            vocab_size, cfg['context_length'],
            embed_dim=cfg['embed_dim'], n_heads=cfg['n_heads'], n_layers=n_layers
        )
        print(f"Parameters: {model.num_parameters:,}")
        
        history = train_model(
            model, *loaders, cfg['epochs'], cfg['lr'], DEVICE,
            is_seq2seq=True,
            save_path=str(RESULTS_DIR / f"transformer_L{n_layers}.pt")
        )
        
        name = f"transformer_L{n_layers}"
        save_history(history, name)
        results[name] = history
        
        plot_training_loss(history, f"Transformer ({n_layers} layers)",
                          str(FIGURES_DIR / f"{name}_training.png"))
    
    print(f"\nTransformer experiments complete")
    return results


def generate_samples(vocab_size: int):
    """Generate text samples from trained models."""
    print("\n" + "=" * 50)
    print("GENERATING TEXT SAMPLES")
    print("=" * 50)
    
    prompt = "HAMLET:"
    train_ds, _, _ = load_shakespeare(str(DATA_DIR), 128, for_transformer=True)
    
    # Load best transformer
    model_path = RESULTS_DIR / "transformer_L4.pt"
    if model_path.exists():
        model = TransformerLM(vocab_size, 128, embed_dim=128, n_heads=4, n_layers=4)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        
        sample = generate_text_char(model, train_ds, prompt, max_length=200, device=DEVICE)
        print(f"\nTransformer sample:\n{sample}")
        
        with open(RESULTS_DIR / "samples.txt", 'w') as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Transformer output:\n{sample}\n")


def generate_summary_plots():
    """Generate summary comparison plots."""
    print("\n" + "=" * 50)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 50)
    
    histories = load_all_histories(str(RESULTS_DIR))
    if not histories:
        print("No histories found!")
        return
    
    plot_architecture_comparison(histories, str(FIGURES_DIR / "architecture_comparison.png"))
    plot_flops_trajectory(histories, "Log-Likelihood vs Training FLOPs",
                         str(FIGURES_DIR / "flops_trajectory.png"))
    plot_model_comparison_bar(histories, 'test_loglik', 'Test Log-Likelihood',
                             str(FIGURES_DIR / "model_comparison.png"))
    
    print("Summary plots generated")


def main():
    parser = argparse.ArgumentParser(description='Train language models')
    parser.add_argument('--model', choices=['linear', 'mlp', 'attention', 'transformer', 'all'],
                       default='all', help='Model to train')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()
    
    # Setup
    setup_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("TINY LANGUAGE MODELS - TRAINING")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get vocab size
    train_ds, _, _ = load_shakespeare(str(DATA_DIR), 64)
    vocab_size = train_ds.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Run experiments
    if args.model in ['linear', 'all']:
        run_linear_experiments(vocab_size, args.quick)
    if args.model in ['mlp', 'all']:
        run_mlp_experiments(vocab_size, args.quick)
    if args.model in ['attention', 'all']:
        run_attention_experiments(vocab_size, args.quick)
    if args.model in ['transformer', 'all']:
        run_transformer_experiments(vocab_size, args.quick)
    
    # Generate samples and plots
    if args.model == 'all':
        generate_samples(vocab_size)
    
    if not args.skip_plots:
        generate_summary_plots()
    
    print("\n" + "=" * 50)
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
