"""Linear (softmax regression) language model baseline."""

import torch
import torch.nn as nn


class LinearPredictor(nn.Module):
    """
    Simple linear baseline for next-token prediction.
    
    Flattens the context window and applies a single linear layer
    (equivalent to softmax regression on one-hot encoded input).
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Number of tokens in the context window
        
    Example:
        >>> model = LinearPredictor(vocab_size=65, context_length=64)
        >>> x = torch.randint(0, 65, (32, 64))  # batch of 32, context 64
        >>> logits = model(x)  # (32, 65)
    """
    
    def __init__(self, vocab_size: int, context_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # Embedding acts like one-hot encoding
        self.embed = nn.Embedding(vocab_size, vocab_size)
        self.linear = nn.Linear(vocab_size * context_length, vocab_size)
        
        # Initialize embedding to identity for interpretability
        nn.init.eye_(self.embed.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices, shape (batch_size, context_length)
            
        Returns:
            Logits for next token, shape (batch_size, vocab_size)
        """
        batch_size = x.size(0)
        embedded = self.embed(x)  # (batch, context_length, vocab_size)
        flattened = embedded.view(batch_size, -1)  # (batch, context_length * vocab_size)
        logits = self.linear(flattened)  # (batch, vocab_size)
        return logits
    
    def count_flops(self, batch_size: int = 1) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        Args:
            batch_size: Batch size for computation
            
        Returns:
            Approximate number of floating point operations
        """
        input_dim = self.vocab_size * self.context_length
        # Linear layer: 2 * input_dim * output_dim (multiply-add)
        return batch_size * 2 * input_dim * self.vocab_size
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
