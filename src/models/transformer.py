"""Full Transformer language model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with attention and feed-forward network.
    
    Uses pre-norm architecture (LayerNorm before attention/FFN)
    with residual connections.
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        context_length: Maximum sequence length
        mlp_ratio: FFN hidden dimension multiplier
        dropout: Dropout probability
        
    Example:
        >>> block = TransformerBlock(embed_dim=128, n_heads=4, context_length=64)
        >>> x = torch.randn(32, 64, 128)
        >>> out = block(x)  # (32, 64, 128)
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        context_length: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Full Transformer language model for autoregressive text generation.
    
    Features:
    - Stacked transformer blocks with attention + FFN
    - Pre-norm architecture for stable training
    - Weight tying between embedding and output layers
    - Configurable depth, width, and attention heads
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        mlp_ratio: FFN hidden dimension = embed_dim * mlp_ratio
        dropout: Dropout probability
        
    Example:
        >>> model = TransformerLM(vocab_size=65, context_length=128)
        >>> x = torch.randint(0, 65, (32, 128))
        >>> logits = model(x)  # (32, 65) - next token prediction
        >>> logits_all = model(x, return_all=True)  # (32, 128, 65) - all positions
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        
        # Token and position embeddings
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, context_length, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        self.tok_embed.weight = self.lm_head.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices, shape (batch_size, seq_len)
            return_all: If True, return logits for all positions;
                       if False, only return logits for last position
            
        Returns:
            Logits, shape (batch_size, vocab_size) or 
                    (batch_size, seq_len, vocab_size) if return_all=True
        """
        B, T = x.size()
        
        # Embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.tok_embed(x) + self.pos_embed(positions))
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        h = self.ln_f(h)
        
        if return_all:
            logits = self.lm_head(h)  # (B, T, vocab_size)
        else:
            logits = self.lm_head(h[:, -1, :])  # (B, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices, shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: If set, only sample from top-k most likely tokens
            
        Returns:
            Generated sequence, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Get predictions
            logits = self.forward(idx_cond, return_all=True)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def count_flops(self, batch_size: int = 1) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        Args:
            batch_size: Batch size for computation
            
        Returns:
            Approximate number of floating point operations
        """
        flops = 0
        T = self.context_length
        d = self.embed_dim
        ff_dim = int(d * self.mlp_ratio)
        
        for _ in range(self.n_layers):
            # Attention
            flops += 3 * 2 * T * d * d  # QKV projection
            flops += 2 * T * T * d      # Attention scores
            flops += 2 * T * d * d      # Output projection
            
            # FFN
            flops += 2 * T * d * ff_dim  # Up projection
            flops += 2 * T * ff_dim * d  # Down projection
        
        # LM head
        flops += 2 * d * self.vocab_size
        
        return batch_size * flops
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
