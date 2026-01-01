#!/usr/bin/env python3
"""
Generate text samples from trained language models.

Examples:
    python scripts/generate.py --prompt "HAMLET:"
    python scripts/generate.py --model transformer --length 500
    python scripts/generate.py --temperature 0.5 --top-k 50
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import LinearPredictor, MLPLanguageModel, AttentionLanguageModel, TransformerLM
from data import load_shakespeare, load_ptb, load_wikitext
from training import generate_text_char, generate_text_word

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_type: str, vocab_size: int, context_length: int):
    """Load a trained model from checkpoint."""
    if model_type == 'linear':
        model = LinearPredictor(vocab_size, context_length)
        path = RESULTS_DIR / f"linear_ctx{context_length}.pt"
    elif model_type == 'mlp':
        model = MLPLanguageModel(vocab_size, context_length, embed_dim=32, 
                                 hidden_dims=[256, 256, 256])
        path = RESULTS_DIR / "mlp_256x256x256.pt"
    elif model_type == 'attention':
        model = AttentionLanguageModel(vocab_size, context_length, 
                                       embed_dim=128, n_heads=4, n_layers=2)
        path = RESULTS_DIR / "attention_h4.pt"
    elif model_type == 'transformer':
        model = TransformerLM(vocab_size, context_length,
                             embed_dim=128, n_heads=4, n_layers=4)
        path = RESULTS_DIR / "transformer_L4.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded checkpoint: {path}")
    else:
        print(f"Warning: No checkpoint found at {path}")
        print("Using randomly initialized model")
    
    return model.to(DEVICE)


def main():
    parser = argparse.ArgumentParser(description='Generate text from language models')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['linear', 'mlp', 'attention', 'transformer'],
                       help='Model architecture to use')
    parser.add_argument('--prompt', type=str, default='HAMLET:',
                       help='Starting prompt for generation')
    parser.add_argument('--length', type=int, default=200,
                       help='Number of characters/words to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                       choices=['shakespeare', 'ptb', 'wikitext'],
                       help='Dataset (determines tokenization)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("TEXT GENERATION")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")
    print()
    
    # Determine context length based on model
    context_length = 128 if args.model == 'transformer' else 64
    
    # Load dataset for vocabulary
    if args.dataset == 'shakespeare':
        train_ds, _, _ = load_shakespeare(str(DATA_DIR), context_length, 
                                          for_transformer=(args.model == 'transformer'))
        vocab_size = train_ds.vocab_size
        generate_fn = generate_text_char
    else:
        if args.dataset == 'ptb':
            train_ds, _, _ = load_ptb(str(DATA_DIR), context_length, for_transformer=True)
        else:
            train_ds, _, _ = load_wikitext(str(DATA_DIR), context_length, for_transformer=True)
        vocab_size = train_ds.vocab_size
        generate_fn = generate_text_word
    
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Load model
    model = load_model(args.model, vocab_size, context_length)
    
    # Generate
    print("-" * 50)
    print("GENERATED TEXT:")
    print("-" * 50)
    
    output = generate_fn(
        model, train_ds, args.prompt,
        max_length=args.length,
        temperature=args.temperature,
        device=DEVICE
    )
    
    print(output)
    print("-" * 50)


if __name__ == '__main__':
    main()
