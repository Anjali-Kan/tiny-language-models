"""Self-attention language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.
    
    Implements scaled dot-product attention with causal masking
    to prevent attending to future tokens.
    
    Args:
        embed_dim: Total embedding dimension
        n_heads: Number of attention heads
        context_length: Maximum sequence length (for causal mask)
        dropout: Dropout probability
        
    Example:
        >>> attn = CausalSelfAttention(embed_dim=128, n_heads=4, context_length=64)
        >>> x = torch.randn(32, 64, 128)  # (batch, seq, embed)
        >>> out = attn(x)  # (32, 64, 128)
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert embed_dim % n_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer('mask', mask.view(1, 1, context_length, context_length))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal self-attention.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor, shape (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.size()
        
        # Compute Q, K, V in one projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :T, :T] == 0, 
            float('-inf')
        )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ v
        
        # Reshape back: (B, T, embed_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        
        return out


class AttentionLanguageModel(nn.Module):
    """
    Language model using only self-attention (no FFN blocks).
    
    Stacks multiple self-attention layers with residual connections
    and layer normalization (pre-norm style).
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of attention layers
        dropout: Dropout probability
        
    Example:
        >>> model = AttentionLanguageModel(vocab_size=65, context_length=64)
        >>> x = torch.randint(0, 65, (32, 64))
        >>> logits = model(x)  # (32, 65)
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Embeddings
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Attention layers with layer norm
        self.attn_layers = nn.ModuleList([
            CausalSelfAttention(embed_dim, n_heads, context_length, dropout)
            for _ in range(n_layers)
        ])
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices, shape (batch_size, seq_len)
            
        Returns:
            Logits for next token, shape (batch_size, vocab_size)
        """
        B, T = x.size()
        
        # Token + positional embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.tok_embed(x) + self.pos_embed(positions))
        
        # Apply attention layers with residual connections
        for attn, ln in zip(self.attn_layers, self.ln_layers):
            h = h + attn(ln(h))
        
        h = self.ln_final(h)
        
        # Only predict from last position
        logits = self.head(h[:, -1, :])
        
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
        T = self.context_length
        d = self.embed_dim
        
        for _ in range(self.n_layers):
            flops += 3 * 2 * T * d * d  # QKV projection
            flops += 2 * T * T * d      # Attention scores
            flops += 2 * T * d * d      # Output projection
        
        flops += 2 * d * self.vocab_size  # LM head
        
        return batch_size * flops
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
