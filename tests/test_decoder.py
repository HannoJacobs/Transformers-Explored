"""Tests for decoder model"""

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
from src.decoder_only import (
    CustomDecoderLayer,
    TransformerModel,
)  # pylint: disable=C0413

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")


class TestDecoder:
    def test_custom_decoder_layer_output_shape(self):
        d_model = 16
        nhead = 4
        dim_feedforward = 32
        dropout = 0.0
        seq_len = 5
        batch_size = 2

        layer = CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        x = torch.randn(seq_len, batch_size, d_model, device=DEVICE)
        # Causal mask
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        attn_mask.fill_diagonal_(0.0)
        # No padding
        key_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=DEVICE)

        out = layer(x, attn_mask, key_pad_mask)
        assert out.shape == (seq_len, batch_size, d_model)

    def test_transformer_model_output_shape(self):
        vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        seq_len = 7
        batch_size = 3
        model = TransformerModel(vocab).to(DEVICE)
        x = torch.randint(0, len(vocab), (seq_len, batch_size), device=DEVICE)
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        attn_mask.fill_diagonal_(0.0)
        key_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=DEVICE)

        out = model(x, attn_mask, key_pad_mask)
        assert out.shape == (seq_len, batch_size, len(vocab))

    def test_transformer_model_handles_padding(self):
        vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        pad_id = vocab["<pad>"]
        seq_len = 6
        batch_size = 2
        model = TransformerModel(vocab).to(DEVICE)
        x = torch.full((seq_len, batch_size), pad_id, dtype=torch.long, device=DEVICE)
        # Set some tokens to non-pad
        x[0, 0] = vocab["foo"]
        x[1, 1] = vocab["bar"]
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        attn_mask.fill_diagonal_(0.0)
        key_pad_mask = (x == pad_id).T  # (B, T)

        out = model(x, attn_mask, key_pad_mask)
        assert out.shape == (seq_len, batch_size, len(vocab))


if __name__ == "__main__":
    pytest.main([__file__])
