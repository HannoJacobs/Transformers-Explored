# pylint: disable=E0633,E1102
import math
import torch
from torch import nn


class MHA(nn.Module):
    """
    A from-scratch implementation of Multi-Head Attention for learning.

    This module is a corrected drop-in replacement for nn.MultiheadAttention
    when `batch_first=False`. It assumes input tensors are of the shape
    (Seq_Len, Batch_Size, Embed_Dim).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # A single large linear layer for Q, K, V projections, just like in the
        # optimized `nn.MultiheadAttention` implementation.
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)

        # The final output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
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
        Shapes:
            - query: (L, B, E) - Target sequence length, Batch size, Embed_dim
            - key:   (S, B, E) - Source sequence length, Batch size, Embed_dim
            - value: (S, B, E) - Source sequence length, Batch size, Embed_dim
        """
        seq_len_q, batch_size, _ = query.shape
        seq_len_kv = key.shape[0]

        # 1. Combined Linear Projection
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
            q = nn.functional.linear(query, w_q, b_q)
            k = nn.functional.linear(key, w_k, b_k)
            v = nn.functional.linear(value, w_v, b_v)

        # 2. Reshape for Multi-Head Computation
        q = q.view(seq_len_q, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )
        k = k.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )
        v = v.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # 4. Concatenate Heads and Project
        context = (
            context.permute(2, 0, 1, 3)
            .contiguous()
            .view(seq_len_q, batch_size, self.embed_dim)
        )

        attn_output = self.out_proj(context)

        # `nn.MultiheadAttention` always returns a tuple. When `need_weights=False`,
        # the second element of the tuple is None. We must replicate this behavior
        # for our module to be a drop-in replacement.
        if need_weights:
            # Return average attention weights over heads
            return attn_output, attn_weights.mean(dim=1)
        else:
            # Return None for the weights, but still inside a tuple
            return attn_output, None
