#!/usr/bin/env python3
"""
Interactive text generation with trained language models.

Usage:
    python scripts/interactive.py
    python scripts/interactive.py --model mlp --temperature 0.5
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import TransformerLM, MLPLanguageModel, AttentionLanguageModel, LinearPredictor
from data import load_shakespeare
from training import generate_text_char

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_type: str, vocab_size: int):
    """Load a trained model."""
    if model_type == 'transformer':
        model = TransformerLM(vocab_size, 128, embed_dim=128, n_heads=4, n_layers=4)
        path = RESULTS_DIR / "transformer_L4.pt"
        ctx_len = 128
    elif model_type == 'attention':
        model = AttentionLanguageModel(vocab_size, 64, embed_dim=128, n_heads=4, n_layers=2)
        path = RESULTS_DIR / "attention_h4.pt"
        ctx_len = 64
    elif model_type == 'mlp':
        model = MLPLanguageModel(vocab_size, 64, embed_dim=32, hidden_dims=[256, 256, 256])
        path = RESULTS_DIR / "mlp_256x256x256.pt"
        ctx_len = 64
    else:
        model = LinearPredictor(vocab_size, 64)
        path = RESULTS_DIR / "linear_ctx64.pt"
        ctx_len = 64
    
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        print(f"Loaded: {path.name}")
    else:
        print(f"⚠ No checkpoint found, using random weights")
    
    return model.to(DEVICE), ctx_len


def main():
    parser = argparse.ArgumentParser(description='Interactive text generation')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['linear', 'mlp', 'attention', 'transformer'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--length', type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔤 TINY LANGUAGE MODEL - INTERACTIVE MODE")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load model and dataset
    ctx_len = 128 if args.model == 'transformer' else 64
    train_ds, _, _ = load_shakespeare(str(DATA_DIR), ctx_len, 
                                       for_transformer=(args.model == 'transformer'))
    vocab_size = train_ds.vocab_size
    
    model, _ = load_model(args.model, vocab_size)
    model.eval()
    
    print("Ready! Enter a prompt to generate text.")
    print("Type 'quit' to exit, 'temp X' to change temperature.\n")
    print("-" * 50)
    
    temperature = args.temperature
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if prompt.lower().startswith('temp '):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Usage: temp 0.5")
            continue
        
        # Generate
        print("\n🤖 Generated:")
        print("-" * 30)
        
        output = generate_text_char(
            model, train_ds, prompt,
            max_length=args.length,
            temperature=temperature,
            device=DEVICE
        )
        print(output)
        print("-" * 30)


if __name__ == '__main__':
    main()