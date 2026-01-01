"""Multi-layer perceptron language model."""

from typing import List, Literal

import torch
import torch.nn as nn


class MLPLanguageModel(nn.Module):
    """
    Multi-layer perceptron for next-token prediction.
    
    Embeds tokens, flattens the context, and passes through multiple
    hidden layers with nonlinear activations.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Number of tokens in the context window
        embed_dim: Dimension of token embeddings
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'tanh')
        dropout: Dropout probability
        
    Example:
        >>> model = MLPLanguageModel(
        ...     vocab_size=65, 
        ...     context_length=64,
        ...     hidden_dims=[256, 256, 256]
        ... )
        >>> x = torch.randint(0, 65, (32, 64))
        >>> logits = model(x)  # (32, 65)
    """
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
    }
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 64,
        hidden_dims: List[int] = None,
        activation: Literal['relu', 'gelu', 'tanh'] = 'gelu',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Get activation class
        act_fn = self.ACTIVATIONS.get(activation, nn.ReLU)
        
        # Build MLP layers
        layers = []
        input_size = context_length * embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_size, hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_dim
        
        # Output projection
        layers.append(nn.Linear(input_size, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices, shape (batch_size, context_length)
            
        Returns:
            Logits for next token, shape (batch_size, vocab_size)
        """
        batch_size = x.size(0)
        embedded = self.embedding(x)  # (batch, context_length, embed_dim)
        flattened = embedded.view(batch_size, -1)  # (batch, context_length * embed_dim)
        logits = self.mlp(flattened)  # (batch, vocab_size)
        return logits
    
    def count_flops(self, batch_size: int = 1) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        Args:
            batch_size: Batch size for computation
            
        Returns:
            Approximate number of floating point operations
        """
        flops = 0
        input_size = self.context_length * self.embed_dim
        
        for hidden_dim in self.hidden_dims:
            flops += 2 * input_size * hidden_dim  # Linear layer
            input_size = hidden_dim
        
        flops += 2 * input_size * self.vocab_size  # Output layer
        
        return batch_size * flops
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
