"""Training utilities for language models."""

import time
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_seq2seq: bool = False,
    clip_grad: float = 1.0,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        is_seq2seq: Whether to compute loss over all positions
        clip_grad: Gradient clipping value (0 to disable)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        if is_seq2seq:
            # Transformer with sequence output
            logits = model(batch_x, return_all=True)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(B * T, V), batch_y.view(-1))
        else:
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
        
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_seq2seq: bool = False,
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        is_seq2seq: Whether model outputs sequences
        
    Returns:
        Tuple of (average_loss, log_likelihood)
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        if is_seq2seq:
            logits = model(batch_x, return_all=True)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(B * T, V), batch_y.view(-1), reduction='sum')
            n_samples += B * T
        else:
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y, reduction='sum')
            n_samples += batch_x.size(0)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / n_samples
    log_likelihood = -avg_loss
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, log_likelihood, perplexity


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    is_seq2seq: bool = False,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full training loop with validation and early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        test_loader: Test data loader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        is_seq2seq: Whether to compute loss over all positions
        patience: Early stopping patience (epochs without improvement)
        save_path: Path to save best model checkpoint
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history:
        - train_loss: List of training losses per epoch
        - valid_loss: List of validation losses per epoch
        - valid_loglik: List of validation log-likelihoods per epoch
        - test_loss: Final test loss
        - test_loglik: Final test log-likelihood
        - total_flops: Total training FLOPs
        - flops_per_epoch: Cumulative FLOPs at each epoch
        - best_epoch: Epoch with best validation loss
        - train_time: Total training time in seconds
    """
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    
    # History tracking
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_loglik': [],
        'test_loglik': None,
        'test_loss': None,
        'total_flops': 0,
        'flops_per_epoch': [],
        'best_epoch': 0,
    }
    
    # Estimate FLOPs per epoch
    n_batches = len(train_loader)
    batch_size = train_loader.batch_size or 32
    flops_per_batch = model.count_flops(batch_size)
    flops_per_epoch = flops_per_batch * n_batches * 3  # ~3x for backward pass
    
    best_valid_loss = float('inf')
    bad_epochs = 0
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, 
            is_seq2seq, clip_grad=1.0
        )
        
        # Validate
        valid_loss, valid_ll, perplexity = evaluate(model, valid_loader, device, is_seq2seq)
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_loglik'].append(valid_ll)
        history['total_flops'] += flops_per_epoch
        history['flops_per_epoch'].append(history['total_flops'])
        
        epoch_time = time.time() - epoch_start
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Valid: {valid_loss:.4f} | "
                f"LL: {valid_ll:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            history['best_epoch'] = epoch
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
    
    total_time = time.time() - start_time
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    # Final test evaluation
    test_loss, test_ll,test_perplexity = evaluate(model, test_loader, device, is_seq2seq)
    history['test_loss'] = test_loss
    history['test_loglik'] = test_ll
    history['train_time'] = total_time
    
    if verbose:
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Test Loss: {test_loss:.4f} | Test LogLik: {test_ll:.4f}")
    
    # Save best model
    if save_path and best_state is not None:
        torch.save(best_state, save_path)
    
    return history


@torch.no_grad()
def generate_text_char(
    model: nn.Module,
    dataset,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    device: torch.device = torch.device('cuda'),
) -> str:
    """
    Generate text character by character.
    
    Args:
        model: Trained language model
        dataset: Dataset with encode/decode methods
        prompt: Starting text
        max_length: Number of characters to generate
        temperature: Sampling temperature
        device: Device to run generation on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    context = dataset.encode(prompt)
    context_length = model.context_length if hasattr(model, 'context_length') else 64
    
    # Pad or truncate
    if len(context) < context_length:
        context = [0] * (context_length - len(context)) + context
    else:
        context = context[-context_length:]
    
    generated = list(prompt)
    
    for _ in range(max_length):
        x = torch.tensor([context], dtype=torch.long, device=device)
        
        if hasattr(model, 'generate'):
            # Transformer with generate method
            out = model.generate(x, 1, temperature=temperature)
            next_idx = out[0, -1].item()
        else:
            # Other models - forward pass
            logits = model(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs[0], 1).item()
        
        generated.append(dataset.idx2char.get(next_idx, '?'))
        context = context[1:] + [next_idx]
    
    return ''.join(generated)


@torch.no_grad()
def generate_text_word(
    model: nn.Module,
    dataset,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    device: torch.device = torch.device('cuda'),
) -> str:
    """
    Generate text word by word.
    
    Args:
        model: Trained language model
        dataset: Dataset with encode/decode methods
        prompt: Starting text
        max_length: Number of words to generate
        temperature: Sampling temperature
        device: Device to run generation on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    context = dataset.encode(prompt)
    context_length = model.context_length if hasattr(model, 'context_length') else 64
    
    if len(context) < context_length:
        context = [0] * (context_length - len(context)) + context
    else:
        context = context[-context_length:]
    
    generated = prompt.split()
    
    for _ in range(max_length):
        x = torch.tensor([context], dtype=torch.long, device=device)
        
        if hasattr(model, 'generate'):
            out = model.generate(x, 1, temperature=temperature)
            next_idx = out[0, -1].item()
        else:
            logits = model(x)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs[0], 1).item()
        
        word = dataset.idx2word.get(next_idx, '<unk>')
        generated.append(word)
        context = context[1:] + [next_idx]
    
    return ' '.join(generated)
