# pylint: disable=E0633,E1102
import math
import torch
from torch import nn

################################################################################
# Multi-Head Attention (MHA) - Educational Implementation
#
# This file provides a from-scratch, annotated implementation of Multi-Head
# Attention (MHA), a core component of the Transformer architecture.
#
# MHA allows the model to jointly attend to information from different
# representation subspaces at different positions. It is widely used in
# natural language processing (NLP) and other domains.
#
# This implementation is designed for learning and understanding, and closely
# follows the interface of PyTorch's `nn.MultiheadAttention` (with batch_first=False).
################################################################################


class MHA(nn.Module):
    """
    Multi-Head Attention (MHA) module - from scratch, for learning.

    This class implements the core logic of MHA as described in the "Attention
    is All You Need" paper (Vaswani et al., 2017).

    Key Features:
    - Drop-in replacement for `nn.MultiheadAttention` (when `batch_first=False`)
    - Assumes input tensors are shaped (Seq_Len, Batch_Size, Embed_Dim)
    - Handles both self-attention and cross-attention

    Args:
        embed_dim (int): Total dimension of the model (input embedding size)
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability on attention weights

    Example:
        mha = MHA(embed_dim=512, num_heads=8)
        output, attn_weights = mha(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimensionality per head

        # A single large linear layer for Q, K, V projections.
        # This is more efficient than separate layers and matches PyTorch's implementation.
        # Input: (Seq_Len, Batch_Size, Embed_Dim)
        # Output: (Seq_Len, Batch_Size, 3 * Embed_Dim)
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)

        # Final output projection layer: projects concatenated heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer for regularizing attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor for dot-product attention (see Vaswani et al.)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
    ):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query: (L, B, E) - Target sequence length, Batch size, Embed_dim
            key:   (S, B, E) - Source sequence length, Batch size, Embed_dim
            value: (S, B, E) - Source sequence length, Batch size, Embed_dim
            attn_mask: Optional mask to prevent attention to certain positions (e.g., future tokens)
            key_padding_mask: Optional mask to ignore padding tokens in the key
            need_weights: If True, also return average attention weights

        Returns:
            attn_output: (L, B, E) - Output of the attention layer
            attn_weights: (B, L, S) or None - Average attention weights over heads (if requested)
        """
        # Unpack input shapes for clarity
        seq_len_q, batch_size, _ = query.shape  # L, B, E
        seq_len_kv = key.shape[0]  # S

        # 1. Combined Linear Projection for Q, K, V
        # If query, key, and value are the same tensor (self-attention), we can
        # project them together for efficiency.
        if torch.equal(query, key) and torch.equal(key, value):
            # in_proj returns (L, B, 3*E); chunk into Q, K, V along the last dim
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:
            # For cross-attention, project Q, K, V separately using the same weights
            w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
            q = nn.functional.linear(query, w_q, b_q)
            k = nn.functional.linear(key, w_k, b_k)
            v = nn.functional.linear(value, w_v, b_v)

        # 2. Reshape for Multi-Head Computation
        # We want to split the embedding into multiple heads for parallel attention.
        # New shape: (Batch, Num_Heads, Seq_Len, Head_Dim)
        q = q.view(seq_len_q, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )  # (B, H, L, D)
        k = k.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )  # (B, H, S, D)
        v = v.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )  # (B, H, S, D)

        # 3. Scaled Dot-Product Attention
        # Compute attention scores: (B, H, L, D) x (B, H, D, S) -> (B, H, L, S)
        # Each query vector attends to all key vectors.
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Optionally add an attention mask (e.g., for causal or padding masking)
        if attn_mask is not None:
            # attn_mask should be broadcastable to (B, H, L, S)
            scores = scores + attn_mask

        # Optionally mask out padding tokens in the key
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> (B, 1, 1, S) for broadcasting
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Softmax over the last dimension (S: source sequence length)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # Regularization

        # Weighted sum of value vectors, using attention weights
        # (B, H, L, S) x (B, H, S, D) -> (B, H, L, D)
        context = torch.matmul(attn_weights, v)

        # 4. Concatenate Heads and Project
        # Rearrange and merge heads: (B, H, L, D) -> (L, B, H*D=E)
        context = (
            context.permute(2, 0, 1, 3)  # (L, B, H, D)
            .contiguous()
            .view(seq_len_q, batch_size, self.embed_dim)
        )

        # Final output projection: (L, B, E) -> (L, B, E)
        attn_output = self.out_proj(context)

        # Return output and (optionally) average attention weights over heads
        if need_weights:
            # Average over heads: (B, H, L, S) -> (B, L, S)
            return attn_output, attn_weights.mean(dim=1)
        else:
            # Return None for the weights, but still inside a tuple (for API compatibility)
            return attn_output, None


################################################################################
# Summary of Key Concepts:
#
# - Q, K, V: Query, Key, Value projections of the input(s)
# - Multi-Head: Split embedding into multiple "heads" for parallel attention
# - Scaled Dot-Product: Compute attention scores, scale by sqrt(head_dim)
# - Masking: Prevent attention to certain positions (e.g., padding, future tokens)
# - Output: Concatenate heads, project back to original embedding size
#
# This implementation is intended for educational purposes and clarity.
################################################################################
