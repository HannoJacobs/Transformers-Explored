# pylint: disable=E0633,E1102
import math
import torch
from torch import nn


class MHA(nn.Module):
    """
    Key Features:
    - Drop-in replacement for `nn.MultiheadAttention` (when `batch_first=False`)
    - Assumes input tensors are shaped (Seq_Len, Batch_Size, Embed_Dim)
    - Handles both self-attention and cross-attention

    Args:
        embed_dim (int): Total dimension of the model (input embedding size)
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability on attention weights
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
        if torch.equal(query, key) and torch.equal(key, value):  # self-attention
            # in_proj returns (L, B, 3*E); chunk into Q, K, V along the last dim
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:  # For cross-attention, project Q, K, V separately using the same weights
            w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
            q = nn.functional.linear(query, w_q, b_q)
            k = nn.functional.linear(key, w_k, b_k)
            v = nn.functional.linear(value, w_v, b_v)

        # 2. Reshape for Multi-Head Computation
        # We want to split the embedding into multiple heads for parallel attention.
        # Initial shape: (Seq_Len, Batch_Size, Embed_Dim)
        # New shape after view+permute: (Batch, Num_Heads, Seq_Len, Head_Dim)
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
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Add an attention mask (e.g., for causal or padding masking)
        if attn_mask is not None:
            # attn_mask should be broadcastable to (B, H, L, S)
            attention_scores = attention_scores + attn_mask

        # mask the parts of the sequences in the batches that are shorter than max len
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) -> (B, 1, 1, S) for broadcasting
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # 4. Softmax over the last dimension (S: source sequence length)
        attn_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # Regularization

        # Weighted sum of value vectors, using attention weights
        # (B, H, L, S) x (B, H, S, D) -> (B, H, L, D)
        context = torch.matmul(attn_weights, v)

        # 5. Concatenate Heads and Project
        # Rearrange and merge heads: (B, H, L, D) -> (L, B, H*D=E)
        context = (
            context.permute(2, 0, 1, 3)  # (L, B, H, D)
            .contiguous()
            .view(seq_len_q, batch_size, self.embed_dim)
        )

        # 6. Final output projection: (L, B, E) -> (L, B, E)
        attn_output = self.out_proj(context)

        # Return output and (optionally) average attention weights over heads
        if need_weights:
            # Average over heads: (B, H, L, S) -> (B, L, S)
            return attn_output, attn_weights.mean(dim=1)
        else:
            # Return None for the weights, but still inside a tuple (for API compatibility)
            return attn_output, None
