"""Tests for encoder-decoder model"""

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
from src.encoder_decoder import (
    CustomDecoderLayer,
    CustomEncoderLayer,
    TransformerModel,
)  # pylint: disable=C0413

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")


class TestEncoderDecoder:
    def test_custom_encoder_layer_output_shape(self):
        d_model = 16
        nhead = 4
        dim_feedforward = 32
        dropout = 0.0
        seq_len = 5
        batch_size = 2

        layer = CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        x = torch.randn(seq_len, batch_size, d_model, device=DEVICE)
        key_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=DEVICE)
        out = layer(x, key_pad_mask)
        assert out.shape == (seq_len, batch_size, d_model)

    def test_custom_decoder_layer_output_shape(self):
        d_model = 16
        nhead = 4
        dim_feedforward = 32
        dropout = 0.0
        tgt_len = 6
        src_len = 5
        batch_size = 2

        layer = CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        tgt = torch.randn(tgt_len, batch_size, d_model, device=DEVICE)
        memory = torch.randn(src_len, batch_size, d_model, device=DEVICE)
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        tgt_key_pad_mask = torch.zeros(
            batch_size, tgt_len, dtype=torch.bool, device=DEVICE
        )
        memory_key_pad_mask = torch.zeros(
            batch_size, src_len, dtype=torch.bool, device=DEVICE
        )
        out = layer(tgt, memory, tgt_mask, tgt_key_pad_mask, memory_key_pad_mask)
        assert out.shape == (tgt_len, batch_size, d_model)

    def test_transformer_model_output_shape(self):
        src_vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        tgt_vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        src_len = 7
        tgt_len = 6
        batch_size = 3
        model = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
        src = torch.randint(0, len(src_vocab), (src_len, batch_size), device=DEVICE)
        src_pad = torch.zeros(batch_size, src_len, dtype=torch.bool, device=DEVICE)
        tgt_in = torch.randint(0, len(tgt_vocab), (tgt_len, batch_size), device=DEVICE)
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        tgt_pad = torch.zeros(batch_size, tgt_len, dtype=torch.bool, device=DEVICE)
        out = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        assert out.shape == (tgt_len, batch_size, len(tgt_vocab))

    def test_transformer_model_handles_padding(self):
        src_vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        tgt_vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        pad_id_src = src_vocab["<pad>"]
        pad_id_tgt = tgt_vocab["<pad>"]
        src_len = 5
        tgt_len = 4
        batch_size = 2
        model = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
        src = torch.full(
            (src_len, batch_size), pad_id_src, dtype=torch.long, device=DEVICE
        )
        tgt_in = torch.full(
            (tgt_len, batch_size), pad_id_tgt, dtype=torch.long, device=DEVICE
        )
        # Set some tokens to non-pad
        src[0, 0] = src_vocab["foo"]
        src[1, 1] = src_vocab["bar"]
        tgt_in[0, 0] = tgt_vocab["hello"]
        tgt_in[1, 1] = tgt_vocab["world"]
        src_pad = (src == pad_id_src).T  # (B, S)
        tgt_pad = (tgt_in == pad_id_tgt).T  # (B, T)
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        out = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        assert out.shape == (tgt_len, batch_size, len(tgt_vocab))

    def test_custom_encoder_layer_respects_padding(self):
        d_model, nhead, dim_feedforward, dropout = 8, 2, 16, 0.0
        seq_len, batch_size = 4, 2
        layer = CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        x = torch.randn(seq_len, batch_size, d_model, device=DEVICE)
        key_pad_mask = torch.tensor(
            [[0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool, device=DEVICE
        )
        out = layer(x, key_pad_mask)
        assert out.shape == (seq_len, batch_size, d_model)
        # Check that padding positions are not all zeros (should be processed, but not attended to)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_custom_encoder_layer_gradient(self):
        d_model, nhead, dim_feedforward, dropout = 8, 2, 16, 0.0
        seq_len, batch_size = 4, 2
        layer = CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        x = torch.randn(seq_len, batch_size, d_model, device=DEVICE, requires_grad=True)
        key_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=DEVICE)
        out = layer(x, key_pad_mask)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_custom_decoder_layer_cross_attention(self):
        d_model, nhead, dim_feedforward, dropout = 8, 2, 16, 0.0
        tgt_len, src_len, batch_size = 5, 4, 2
        layer = CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout).to(DEVICE)
        tgt = torch.randn(tgt_len, batch_size, d_model, device=DEVICE)
        memory = torch.randn(src_len, batch_size, d_model, device=DEVICE)
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        tgt_key_pad_mask = torch.zeros(
            batch_size, tgt_len, dtype=torch.bool, device=DEVICE
        )
        memory_key_pad_mask = torch.zeros(
            batch_size, src_len, dtype=torch.bool, device=DEVICE
        )
        out = layer(tgt, memory, tgt_mask, tgt_key_pad_mask, memory_key_pad_mask)
        assert out.shape == (tgt_len, batch_size, d_model)

    def test_transformer_model_forward_and_grad(self):
        src_vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        tgt_vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        src_len, tgt_len, batch_size = 5, 6, 2
        model = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
        src = torch.randint(0, len(src_vocab), (src_len, batch_size), device=DEVICE)
        src_pad = torch.zeros(batch_size, src_len, dtype=torch.bool, device=DEVICE)
        tgt_in = torch.randint(0, len(tgt_vocab), (tgt_len, batch_size), device=DEVICE)
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        tgt_pad = torch.zeros(batch_size, tgt_len, dtype=torch.bool, device=DEVICE)
        out = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        assert out.shape == (tgt_len, batch_size, len(tgt_vocab))
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_transformer_model_all_padding(self):
        src_vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        tgt_vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        pad_id_src = src_vocab["<pad>"]
        pad_id_tgt = tgt_vocab["<pad>"]
        src_len, tgt_len, batch_size = 5, 4, 2
        model = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
        src = torch.full(
            (src_len, batch_size), pad_id_src, dtype=torch.long, device=DEVICE
        )
        tgt_in = torch.full(
            (tgt_len, batch_size), pad_id_tgt, dtype=torch.long, device=DEVICE
        )
        src_pad = (src == pad_id_src).T
        tgt_pad = (tgt_in == pad_id_tgt).T
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        tgt_mask.fill_diagonal_(0.0)
        out = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        assert out.shape == (tgt_len, batch_size, len(tgt_vocab))

    def test_transformer_model_single_token(self):
        src_vocab = {
            w: i
            for i, w in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", "foo", "bar"])
        }
        tgt_vocab = {
            w: i
            for i, w in enumerate(
                ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"]
            )
        }
        src = torch.tensor([[src_vocab["foo"]]], device=DEVICE)
        src_pad = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)
        tgt_in = torch.tensor([[tgt_vocab["hello"]]], device=DEVICE)
        tgt_mask = torch.zeros(1, 1, device=DEVICE)
        tgt_pad = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)
        model = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
        out = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        assert out.shape == (1, 1, len(tgt_vocab))


if __name__ == "__main__":
    pytest.main([__file__])
