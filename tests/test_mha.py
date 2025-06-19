"""Tests for MHA"""

# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103,E0401,E0611,E1101,C2801,W1203,W0611
import os
import sys
import copy

import pytest
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.mha import MHA  # pylint: disable=C0413

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")


class TestMHA:
    def test_mha_output_shape(self):
        embed_dim = 16
        num_heads = 4
        seq_len = 10
        batch_size = 2

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        x = torch.randn(seq_len, batch_size, embed_dim, device=DEVICE)
        out, attn_weights = mha(x, x, x)
        assert out.shape == (seq_len, batch_size, embed_dim)
        assert attn_weights is None

    def test_mha_attention_weights(self):
        embed_dim = 8
        num_heads = 2
        seq_len = 5
        batch_size = 3

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        x = torch.randn(seq_len, batch_size, embed_dim, device=DEVICE)
        out, attn_weights = mha(x, x, x, need_weights=True)
        assert out.shape == (seq_len, batch_size, embed_dim)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_mha_masking(self):
        embed_dim = 8
        num_heads = 2
        seq_len = 4
        batch_size = 2

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        x = torch.randn(seq_len, batch_size, embed_dim, device=DEVICE)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        out, attn_weights = mha(x, x, x, attn_mask=attn_mask, need_weights=True)
        assert out.shape == (seq_len, batch_size, embed_dim)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_mha_cross_attention(self):
        embed_dim = 8
        num_heads = 2
        tgt_len = 3
        src_len = 5
        batch_size = 2

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        query = torch.randn(tgt_len, batch_size, embed_dim, device=DEVICE)
        key = torch.randn(src_len, batch_size, embed_dim, device=DEVICE)
        value = torch.randn(src_len, batch_size, embed_dim, device=DEVICE)
        out, attn_weights = mha(query, key, value, need_weights=True)
        assert out.shape == (tgt_len, batch_size, embed_dim)
        assert attn_weights.shape == (batch_size, tgt_len, src_len)

    def test_mha_key_padding_mask(self):
        embed_dim = 8
        num_heads = 2
        seq_len = 6
        batch_size = 2

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        x = torch.randn(seq_len, batch_size, embed_dim, device=DEVICE)
        key_padding_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=DEVICE
        )
        key_padding_mask[0, -2:] = True
        out, attn_weights = mha(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=True
        )
        assert out.shape == (seq_len, batch_size, embed_dim)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_mha_gradients(self):
        embed_dim = 8
        num_heads = 2
        seq_len = 5
        batch_size = 3

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
        x = torch.randn(
            seq_len, batch_size, embed_dim, device=DEVICE, requires_grad=True
        )

        out, _ = mha(x, x, x)
        out.sum().backward()

        assert mha.in_proj.weight.grad is not None
        assert mha.out_proj.weight.grad is not None
        assert mha.in_proj.weight.grad.shape == mha.in_proj.weight.shape
        assert mha.out_proj.weight.grad.shape == mha.out_proj.weight.shape

    def test_mha_dropout(self):
        embed_dim = 8
        num_heads = 2
        seq_len = 5
        batch_size = 3

        mha = MHA(embed_dim=embed_dim, num_heads=num_heads, dropout=0.5).to(DEVICE)
        x = torch.randn(seq_len, batch_size, embed_dim, device=DEVICE)

        mha.train()
        out1, _ = mha(x, x, x)
        out2, _ = mha(x, x, x)
        assert not torch.equal(out1, out2)

        mha.eval()
        out3, _ = mha(x, x, x)
        out4, _ = mha(x, x, x)
        assert torch.equal(out3, out4)

    def _test_mha_vs_pytorch(
        self,
        embed_dim,
        num_heads,
        batch_size,
        tgt_len,
        src_len,
        is_cross_attention,
        use_attn_mask,
        use_key_padding_mask,
    ):
        """Helper function to compare MHA with PyTorch's implementation."""
        mha = MHA(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE).eval()
        pytorch_mha = (
            nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=False
            )
            .to(DEVICE)
            .eval()
        )

        mha.in_proj.weight.data.copy_(pytorch_mha.in_proj_weight)
        mha.in_proj.bias.data.copy_(pytorch_mha.in_proj_bias)
        mha.out_proj.weight.data.copy_(pytorch_mha.out_proj.weight)
        mha.out_proj.bias.data.copy_(pytorch_mha.out_proj.bias)

        query = torch.randn(tgt_len, batch_size, embed_dim, device=DEVICE)
        if is_cross_attention:
            key = torch.randn(src_len, batch_size, embed_dim, device=DEVICE)
            value = torch.randn(src_len, batch_size, embed_dim, device=DEVICE)
        else:
            key, value = query, query

        attn_mask = None
        my_attn_mask = None
        if use_attn_mask:
            attn_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=DEVICE) * float("-inf"), diagonal=1
            )
            my_attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        key_padding_mask = None
        if use_key_padding_mask:
            key_padding_mask = torch.zeros(
                batch_size, src_len, dtype=torch.bool, device=DEVICE
            )
            key_padding_mask[0, -2:] = True

        my_out, my_attn = mha(
            query,
            key,
            value,
            attn_mask=my_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        pytorch_out, pytorch_attn = pytorch_mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        assert torch.allclose(my_out, pytorch_out, atol=1e-6), "Outputs do not match"
        assert torch.allclose(
            my_attn, pytorch_attn, atol=1e-6
        ), "Attention weights do not match"

    @pytest.mark.parametrize("use_attn_mask", [True, False])
    @pytest.mark.parametrize("use_key_padding_mask", [True, False])
    def test_mha_vs_pytorch_self_attention(self, use_attn_mask, use_key_padding_mask):
        self._test_mha_vs_pytorch(
            embed_dim=16,
            num_heads=4,
            batch_size=2,
            tgt_len=10,
            src_len=10,
            is_cross_attention=False,
            use_attn_mask=use_attn_mask,
            use_key_padding_mask=use_key_padding_mask,
        )

    @pytest.mark.parametrize("use_attn_mask", [True, False])
    @pytest.mark.parametrize("use_key_padding_mask", [True, False])
    def test_mha_vs_pytorch_cross_attention(self, use_attn_mask, use_key_padding_mask):
        self._test_mha_vs_pytorch(
            embed_dim=16,
            num_heads=4,
            batch_size=2,
            tgt_len=10,
            src_len=12,
            is_cross_attention=True,
            use_attn_mask=use_attn_mask,
            use_key_padding_mask=use_key_padding_mask,
        )


if __name__ == "__main__":
    pytest.main([__file__])
