# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103,E0401,E0611,E1101,C2801,W1203,W0611
"""Tests for MHA"""


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


if __name__ == "__main__":
    t = TestMHA()
    t.test_mha_output_shape()
    t.test_mha_attention_weights()
    t.test_mha_masking()
    t.test_mha_cross_attention()
    t.test_mha_key_padding_mask()
    print("\nAll tests passed!")
