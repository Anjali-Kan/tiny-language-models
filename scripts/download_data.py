#!/usr/bin/env python3
"""
Download and prepare datasets for language modeling experiments.

Datasets:
    - Tiny Shakespeare: Character-level (~1MB)
    - Penn Treebank: Word-level (~5MB)
    - WikiText-2: Word-level (~12MB)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset shakespeare
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'


URLS = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'ptb': 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz',
    'wikitext': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip',
}


def download_file(url: str, path: Path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def prepare_shakespeare():
    """Download and split Tiny Shakespeare dataset."""
    out_dir = DATA_DIR / 'tiny_shakespeare'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    raw_path = out_dir / 'input.txt'
    if not raw_path.exists():
        download_file(URLS['shakespeare'], raw_path)
    
    # Read and split
    with open(raw_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    n = len(text)
    train_text = text[:int(0.8 * n)]
    valid_text = text[int(0.8 * n):int(0.9 * n)]
    test_text = text[int(0.9 * n):]
    
    # Save splits
    for name, content in [('train', train_text), ('valid', valid_text), ('test', test_text)]:
        with open(out_dir / f'{name}.txt', 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Tiny Shakespeare prepared: {out_dir}")
    print(f"  Train: {len(train_text):,} chars")
    print(f"  Valid: {len(valid_text):,} chars")
    print(f"  Test: {len(test_text):,} chars")


def prepare_wikitext():
    """Download and prepare WikiText-2 dataset."""
    out_dir = DATA_DIR / 'wikitext-2'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = DATA_DIR / 'wikitext-2-raw-v1.zip'
    
    if not (out_dir / 'train.txt').exists():
        # Download
        download_file(URLS['wikitext'], zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
        
        # Move files
        raw_dir = DATA_DIR / 'wikitext-2-raw'
        for split in ['train', 'valid', 'test']:
            src = raw_dir / f'wiki.{split}.raw'
            dst = out_dir / f'{split}.txt'
            if src.exists():
                src.rename(dst)
        
        # Cleanup
        if zip_path.exists():
            zip_path.unlink()
        if raw_dir.exists():
            import shutil
            shutil.rmtree(raw_dir)
    
    print(f"WikiText-2 prepared: {out_dir}")


def prepare_ptb():
    """Prepare Penn Treebank (requires manual download due to licensing)."""
    out_dir = DATA_DIR / 'ptb'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if (out_dir / 'train.txt').exists():
        print(f"PTB already exists: {out_dir}")
        return
    
    print("Penn Treebank requires manual setup due to licensing.")
    print("You can use the preprocessed version from:")
    print("  https://github.com/wojzaremba/lstm/tree/master/data")
    print()
    print("Download ptb.train.txt, ptb.valid.txt, ptb.test.txt")
    print(f"and place them in: {out_dir}")
    print()
    print("Or use this command:")
    print(f"  cd {out_dir}")
    print("  wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt -O train.txt")
    print("  wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt -O valid.txt")
    print("  wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt -O test.txt")


def main():
    parser = argparse.ArgumentParser(description='Download language modeling datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'shakespeare', 'ptb', 'wikitext'],
                       help='Dataset to download')
    args = parser.parse_args()
    
    print("=" * 50)
    print("DOWNLOADING DATASETS")
    print("=" * 50)
    print(f"Data directory: {DATA_DIR}")
    print()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ['all', 'shakespeare']:
        prepare_shakespeare()
        print()
    
    if args.dataset in ['all', 'wikitext']:
        prepare_wikitext()
        print()
    
    if args.dataset in ['all', 'ptb']:
        prepare_ptb()
        print()
    
    print("=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == '__main__':
    main()
