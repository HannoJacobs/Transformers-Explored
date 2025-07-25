"""Seq 2 Seq Transformer Model"""

# pylint: disable=C3001,R0914,R0913,R0917
import os
import sys
import re
import math
import datetime
import time

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.mha import MHA  # pylint: disable=C0413

# NUM_ROWS = "full"
NUM_ROWS = 10000
DATA_PATH = f"Datasets/eng_afr/eng_afr_{NUM_ROWS}_rows.csv"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
INPUT_MAX_SEQ_LEN = 10
OUTPUT_MAX_SEQ_LEN = 10
MAX_GEN_LEN = 100
MIN_FREQ = 1
MASK_VAL = float("-inf")
MAX_SEQ_LEN = max(INPUT_MAX_SEQ_LEN, OUTPUT_MAX_SEQ_LEN) + 2  # (+2 for BOS/EOS)
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")


def tokenize(text: str) -> list[str]:
    """Simple word/punctuation tokenizer."""
    _word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return _word_re.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict, dict]:
    """Builds vocab and inverse vocab from texts, filtering by min_freq."""
    frequency: dict[str, int] = {}
    for line in texts:
        for tok in tokenize(line):
            frequency[tok] = frequency.get(tok, 0) + 1
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    for tok in sorted(frequency):
        if frequency[tok] >= min_freq:
            vocab.setdefault(tok, len(vocab))
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab


def encode(tokens: list[str], vocab: dict) -> list[int]:
    """Converts tokens to integer IDs using the vocabulary."""
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]


class TranslationDataset(Dataset):
    """
    (src_ids, tgt_ids) without BOS/EOS/PAD, truncated
    to INPUT_MAX_SEQ_LEN and OUTPUT_MAX_SEQ_LEN
    """

    def __init__(self, df: pd.DataFrame, source_vocab: dict, target_vocab: dict):
        self.data = []
        for src_sentence, tgt_sentence in zip(df["src"], df["target"]):
            src_tokens = tokenize(src_sentence)
            tgt_tokens = tokenize(tgt_sentence)
            src_tokens_encoded = encode(src_tokens[:INPUT_MAX_SEQ_LEN], source_vocab)
            tgt_tokens_encoded = encode(tgt_tokens[:OUTPUT_MAX_SEQ_LEN], target_vocab)
            if src_tokens_encoded and tgt_tokens_encoded:
                self.data.append((src_tokens_encoded, tgt_tokens_encoded))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, src_pad_id, tgt_pad_id, tgt_bos_id, tgt_eos_id):
    """
    Pads and builds:
    src         (S,B)
    src_pad     (B,S)  bool
    tgt_in/out  (T,B)
    tgt_mask    (T,T)  float  0 / -1e9   <- causal
    tgt_pad     (B,T)  bool
    """
    src_seqs, tgt_seqs = zip(*batch)
    S = max(len(s) for s in src_seqs)
    T = max(len(t) for t in tgt_seqs) + 1  # +1 for BOS/EOS

    src = torch.full((S, len(batch)), src_pad_id, dtype=torch.long)
    tgt_in = torch.full((T, len(batch)), tgt_pad_id, dtype=torch.long)
    tgt_out = torch.full((T, len(batch)), tgt_pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[: len(s), i] = torch.tensor(s)
        tin = [tgt_bos_id] + t
        tout = t + [tgt_eos_id]
        tgt_in[: len(tin), i] = torch.tensor(tin)
        tgt_out[: len(tout), i] = torch.tensor(tout)

    src_key_pad = (src == src_pad_id).T  # (B,S)
    tgt_key_pad = (tgt_in == tgt_pad_id).T  # (B,T)

    # ---- additive causal mask (float) ----
    tgt_mask = torch.triu(torch.full((T, T), MASK_VAL), diagonal=1)
    tgt_mask.fill_diagonal_(0.0)  # diag = 0
    return src, src_key_pad, tgt_in, tgt_out, tgt_mask.float(), tgt_key_pad


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input embeddings."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (L,B,E)
        """Adds positional encoding to input tensor."""
        return self.drop(x + self.pe[: x.size(0)])


class CustomEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer (Pre-Norm)
    -------------------------------------------
    This class implements a single layer of the Transformer encoder, as described in
    "Attention is All You Need" (Vaswani et al., 2017). It is designed for clarity and learning.

    Key Components:
    - Multi-Head Self-Attention (MHA): Each token in the input sequence can attend to all
        other tokens, allowing the model to capture contextual relationships.
    - Feedforward Network (FFN): A position-wise nonlinearity that increases model capacity.
    - Layer Normalization (Pre-Norm): Normalizes inputs before each sub-layer for stability.
    - Residual Connections: Add the input of each sub-layer to its output, improving optimization.
    - Dropout: Regularizes the model to prevent overfitting.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        nhead (int): Number of attention heads in MHA.
        dim_feedforward (int): Hidden size of the feedforward network.
        dropout (float): Dropout probability.

    Forward Pass:
        src: (Seq_Len, Batch, D_Model) - Input sequence embeddings.
        src_key_padding_mask: (Batch, Seq_Len) - Mask to ignore padding tokens.

    Returns:
        (Seq_Len, Batch, D_Model) - Output sequence embeddings after attention and FFN.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Multi-Head Self-Attention: Each token attends to all tokens in the input.
        self.self_attn = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Feedforward Network: Two linear layers with a nonlinearity in between.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization (Pre-Norm): Applied before each sub-layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization after each sub-layer.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Nonlinearity for the feedforward network.
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask):
        """
        Forward pass for a single encoder layer.

        1. Self-Attention Block:
            - Apply multi-head self-attention, using a padding mask to ignore padded tokens.
            - Add the result to the input (residual connection).
            - Normalize the result.

        2. Feedforward Block:
            - Apply the feedforward network (linear -> activation -> dropout -> linear).
            - Add the result to the input (residual connection).
            - Normalize the result.

        Args:
            src: (Seq_Len, Batch, D_Model) - Input embeddings.
            src_key_padding_mask: (Batch, Seq_Len) - Mask for padding tokens.

        Returns:
            (Seq_Len, Batch, D_Model) - Output embeddings.
        """
        #### 1. Self-Attention Block ####
        src2, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask, need_weights=False
        )
        src = src + self.dropout1(src2)  # residual + dropout
        src = self.norm1(src)

        #### 2. Feedforward Block ####
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # residual + dropout
        src = self.norm2(src)

        return src


class CustomDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer (Pre-Norm, Encoder-Decoder)
    ------------------------------------------------------------
    This class implements a single layer of the Transformer decoder, as used in
    sequence-to-sequence models for tasks like translation.

    Key Components:
    - Masked Multi-Head Self-Attention (MHA): Each position in the target sequence
        can only attend to itself and previous positions (causal masking), enabling
        autoregressive generation.
    - Cross-Attention (Encoder-Decoder Attention): Each target position can attend
        to all positions in the encoded source sequence, allowing the decoder to
        leverage the encoded input.
    - Feedforward Network (FFN): Position-wise nonlinearity for increased capacity.
    - Layer Normalization (Pre-Norm): Normalizes inputs before each sub-layer.
    - Residual Connections: Add the input of each sub-layer to its output.
    - Dropout: Regularizes the model.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        nhead (int): Number of attention heads in MHA.
        dim_feedforward (int): Hidden size of the feedforward network.
        dropout (float): Dropout probability.

    Forward Pass:
        tgt: (Tgt_Len, Batch, D_Model) - Target sequence embeddings.
        memory: (Src_Len, Batch, D_Model) - Encoder output (source sequence).
        tgt_mask: (Tgt_Len, Tgt_Len) - Causal mask for autoregressive modeling.
        tgt_key_padding_mask: (Batch, Tgt_Len) - Mask for padding tokens in target.
        memory_key_padding_mask: (Batch, Src_Len) - Mask for padding tokens in source.

    Returns:
        (Tgt_Len, Batch, D_Model) - Output sequence embeddings after attention and FFN.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Masked Multi-Head Self-Attention: Each target token attends to itself and previous tokens.
        self.self_attn = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Cross-Attention: Each target token attends to all source tokens (encoder output).
        self.multihead_attn = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Feedforward Network: Two linear layers with a nonlinearity in between.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization (Pre-Norm): Applied before each sub-layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for regularization after each sub-layer.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Nonlinearity for the feedforward network.
        self.activation = nn.ReLU()

    def forward(
        self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
    ):
        """
        Forward pass for a single decoder layer.

        1. Masked Self-Attention Block:
            - Apply multi-head self-attention to the target sequence, using a causal mask
                to prevent attending to future tokens and a padding mask for padded tokens.
            - Add the result to the input (residual connection).
            - Normalize the result.

        2. Cross-Attention Block:
            - Apply multi-head attention where the query is the target sequence and the
                key/value are the encoder output (memory), using a padding mask for the source.
            - Add the result to the input (residual connection).
            - Normalize the result.

        3. Feedforward Block:
            - Apply the feedforward network (linear -> activation -> dropout -> linear).
            - Add the result to the input (residual connection).
            - Normalize the result.

        Args:
            tgt: (Tgt_Len, Batch, D_Model) - Target embeddings.
            memory: (Src_Len, Batch, D_Model) - Encoder output.
            tgt_mask: (Tgt_Len, Tgt_Len) - Causal mask for autoregressive modeling.
            tgt_key_padding_mask: (Batch, Tgt_Len) - Mask for padding tokens in target.
            memory_key_padding_mask: (Batch, Src_Len) - Mask for padding tokens in source.

        Returns:
            (Tgt_Len, Batch, D_Model) - Output embeddings.
        """
        #### 1. Masked Self-Attention Block ####
        tgt2, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout1(tgt2)  # residual + dropout
        tgt = self.norm1(tgt)

        #### 2. Cross-Attention Block ####
        tgt2, _ = self.multihead_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout2(tgt2)  # residual + dropout
        tgt = self.norm2(tgt)

        #### 3. Feedforward Block ####
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)  # residual + dropout
        tgt = self.norm3(tgt)

        return tgt


class TransformerModel(nn.Module):
    """
    Full Encoder-Decoder Transformer Model for Sequence-to-Sequence Tasks
    ---------------------------------------------------------------------
    This class implements the full Transformer architecture for tasks such as
    machine translation, summarization, and other sequence-to-sequence problems.

    Model Architecture:
    - Source Embedding Layer: Maps source token IDs to dense vectors.
    - Target Embedding Layer: Maps target token IDs to dense vectors.
    - Positional Encoding: Adds information about token positions to both source and target.
    - Stack of Encoder Layers: Each layer applies self-attention and a feedforward network to the source.
    - Stack of Decoder Layers: Each layer applies masked self-attention, cross-attention (to encoder output),
        and a feedforward network to the target.
    - Output Projection: Maps the final decoder hidden states to vocabulary logits.

    Args:
        source_vocab (dict): Vocabulary mapping source tokens to integer IDs.
        target_vocab (dict): Vocabulary mapping target tokens to integer IDs.

    Forward Pass:
        src: (Src_Len, Batch) - Source token IDs.
        src_pad: (Batch, Src_Len) - Mask for padding tokens in source.
        tgt_in: (Tgt_Len, Batch) - Target input token IDs (for teacher forcing).
        tgt_mask: (Tgt_Len, Tgt_Len) - Causal mask for autoregressive modeling.
        tgt_pad: (Batch, Tgt_Len) - Mask for padding tokens in target.

    Returns:
        (Tgt_Len, Batch, Target_Vocab_Size) - Logits for each token in the target vocabulary.
    """

    def __init__(self, source_vocab, target_vocab):
        super().__init__()
        self.d_model = D_MODEL

        # Embedding layers for source and target vocabularies.
        self.source_embed = nn.Embedding(
            num_embeddings=len(source_vocab), embedding_dim=D_MODEL, padding_idx=0
        )
        self.target_embed = nn.Embedding(
            num_embeddings=len(target_vocab), embedding_dim=D_MODEL, padding_idx=0
        )

        # Positional encoding for both source and target sequences.
        self.position_encode = PositionalEncoding(
            d_model=D_MODEL, dropout=DROPOUT, max_len=MAX_SEQ_LEN
        )

        # Stack of encoder layers: Each is a CustomEncoderLayer.
        self.encoder_layers = nn.ModuleList(
            [
                CustomEncoderLayer(
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    dim_feedforward=DIM_FEEDFORWARD,
                    dropout=DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        # Stack of decoder layers: Each is a CustomDecoderLayer.
        self.decoder_layers = nn.ModuleList(
            [
                CustomDecoderLayer(
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    dim_feedforward=DIM_FEEDFORWARD,
                    dropout=DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        # Output projection: Maps decoder hidden states to target vocabulary logits.
        self.projection = nn.Linear(in_features=D_MODEL, out_features=len(target_vocab))

    def encode(self, src, src_pad):
        """
        Encodes the source sequence.

        1. Embed the source token IDs.
        2. Add positional encoding.
        3. Pass through the stack of encoder layers.

        Args:
            src: (Src_Len, Batch) - Source token IDs.
            src_pad: (Batch, Src_Len) - Mask for padding tokens in source.

        Returns:
            (Src_Len, Batch, D_Model) - Encoded source sequence (memory).
        """
        x = self.position_encode(self.source_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_pad)
        return x

    def decode(self, mem, src_pad, tgt_in, tgt_mask, tgt_pad):
        """
        Decodes the target sequence, attending to the encoded source.

        1. Embed the target input token IDs.
        2. Add positional encoding.
        3. Pass through the stack of decoder layers, attending to the encoder output.

        Args:
            mem: (Src_Len, Batch, D_Model) - Encoder output (memory).
            src_pad: (Batch, Src_Len) - Mask for padding tokens in source.
            tgt_in: (Tgt_Len, Batch) - Target input token IDs.
            tgt_mask: (Tgt_Len, Tgt_Len) - Causal mask for autoregressive modeling.
            tgt_pad: (Batch, Tgt_Len) - Mask for padding tokens in target.

        Returns:
            (Tgt_Len, Batch, Target_Vocab_Size) - Logits for each token in the target vocabulary.
        """
        y = self.position_encode(self.target_embed(tgt_in) * math.sqrt(self.d_model))
        out = y
        for layer in self.decoder_layers:
            out = layer(
                out,
                mem,
                tgt_mask=tgt_mask.to(y.device),
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad,
            )
        return self.projection(out)  # (T,B,V)

    def forward(self, src, src_pad, tgt_in, tgt_mask, tgt_pad):
        """
        Full forward pass for the encoder-decoder Transformer.

        1. Encode the source sequence.
        2. Decode the target sequence, attending to the encoder output.

        Args:
            src: (Src_Len, Batch) - Source token IDs.
            src_pad: (Batch, Src_Len) - Mask for padding tokens in source.
            tgt_in: (Tgt_Len, Batch) - Target input token IDs.
            tgt_mask: (Tgt_Len, Tgt_Len) - Causal mask for autoregressive modeling.
            tgt_pad: (Batch, Tgt_Len) - Mask for padding tokens in target.

        Returns:
            (Tgt_Len, Batch, Target_Vocab_Size) - Logits for each token in the target vocabulary.
        """
        mem = self.encode(src, src_pad)
        return self.decode(mem, src_pad, tgt_in, tgt_mask, tgt_pad)


def train_epoch(model, loader, optimizer_, loss_criterion_, pad_id):
    """Trains the model for one epoch."""
    model.train()
    tot_loss = tot_batches = 0
    tot_tok = tot_correct = 0
    for src, src_pad, tgt_in, tgt_out, tgt_mask, tgt_pad in loader:
        src, src_pad = src.to(DEVICE), src_pad.to(DEVICE)
        tgt_in, tgt_out = tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        tgt_mask, tgt_pad = tgt_mask.to(DEVICE), tgt_pad.to(DEVICE)

        optimizer_.zero_grad()
        logits = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()

        # ---- accuracy (token level, ignores PAD) ----
        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = tgt_out.ne(pad_id)
            tot_correct += (pred.eq(tgt_out) & mask).sum().item()
            tot_tok += mask.sum().item()

        tot_loss += loss.item()
        tot_batches += 1
    return tot_loss / tot_batches, tot_correct / tot_tok


@torch.no_grad()
def eval_epoch(model, loader, loss_criterion_, pad_id):
    """Evaluates the model for one epoch."""
    model.eval()
    tot_loss = tot_batches = 0
    tot_tok = tot_correct = 0
    for src, src_pad, tgt_in, tgt_out, tgt_mask, tgt_pad in loader:
        src, src_pad = src.to(DEVICE), src_pad.to(DEVICE)
        tgt_in, tgt_out = tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        tgt_mask, tgt_pad = tgt_mask.to(DEVICE), tgt_pad.to(DEVICE)

        logits = model(src, src_pad, tgt_in, tgt_mask, tgt_pad)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        pred = logits.argmax(-1)
        mask = tgt_out.ne(pad_id)
        tot_correct += (pred.eq(tgt_out) & mask).sum().item()
        tot_tok += mask.sum().item()

        tot_loss += loss.item()
        tot_batches += 1
    return tot_loss / tot_batches, tot_correct / tot_tok


def infer(model, sentence, source_vocab, target_vocab, inv_target_vocab):
    """Translates a source sentence using the trained model."""
    model.eval()
    src_ids = encode(tokenize(sentence)[:INPUT_MAX_SEQ_LEN], source_vocab)
    src = torch.tensor(src_ids, device=DEVICE).unsqueeze(1)  # (S,1)
    src_pad = (src == source_vocab[PAD_TOKEN]).T  # (1,S)
    mem = model.encode(src, src_pad)  # cache encoder output once

    tgt_ids = [target_vocab[BOS_TOKEN]]
    for _ in range(MAX_GEN_LEN):
        if len(tgt_ids) - 1 >= OUTPUT_MAX_SEQ_LEN:
            break
        tgt = torch.tensor(tgt_ids, device=DEVICE).unsqueeze(1)  # (T,1)
        tgt_pad = (tgt == target_vocab[PAD_TOKEN]).T
        tgt_mask = torch.triu(
            torch.full((tgt.size(0), tgt.size(0)), MASK_VAL, device=DEVICE), diagonal=1
        )
        tgt_mask.fill_diagonal_(0.0)
        out = model.decode(mem, src_pad, tgt, tgt_mask, tgt_pad)  # <-- reuse `mem`
        next_id = out[-1, 0].argmax().item()
        if next_id == target_vocab[EOS_TOKEN]:
            break
        tgt_ids.append(next_id)

    words = [inv_target_vocab.get(i, UNK_TOKEN) for i in tgt_ids[1:]]
    sent = []
    for w in words:
        if sent and w.isalnum():
            sent.append(" ")
        sent.append(w)
    return "".join(sent)


if __name__ == "__main__":
    start_time = time.time()
    # ---- 1. Load data ----
    DF = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(DF):,} sentence pairs")

    # ---- 2. Vocab ----
    src_vocab, inv_src_vocab = build_vocab(DF["src"], MIN_FREQ)
    tgt_vocab, inv_tgt_vocab = build_vocab(DF["target"], MIN_FREQ)
    print(f"Src vocab: {len(src_vocab):,} | Tgt vocab: {len(tgt_vocab):,}")
    PAD_ID = tgt_vocab[PAD_TOKEN]

    # ---- 3. Dataset / DataLoader ----
    full_ds = TranslationDataset(DF, src_vocab, tgt_vocab)
    TRAIN_SZ = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(
        full_ds,
        [TRAIN_SZ, len(full_ds) - TRAIN_SZ],
        generator=torch.Generator().manual_seed(42),
    )
    collate_func = lambda b: collate(
        b,
        src_pad_id=src_vocab[PAD_TOKEN],
        tgt_pad_id=tgt_vocab[PAD_TOKEN],
        tgt_bos_id=tgt_vocab[BOS_TOKEN],
        tgt_eos_id=tgt_vocab[EOS_TOKEN],
    )
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_func)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_func)

    # ---- 4. Model / Optim / Loss ----
    MODEL = TransformerModel(src_vocab, tgt_vocab).to(DEVICE)
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[PAD_TOKEN])

    # ---- 5. Training loop ----
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_perplexities, val_perplexities = [], []
    epochs_range = range(1, EPOCHS + 1)
    for ep in epochs_range:
        epoch_start_time = time.time()
        tr_loss, tr_acc = train_epoch(
            MODEL, train_dl, optimizer, loss_criterion, PAD_ID
        )
        vl_loss, vl_acc = eval_epoch(MODEL, val_dl, loss_criterion, PAD_ID)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        tr_ppl = math.exp(tr_loss)
        vl_ppl = math.exp(vl_loss)
        train_perplexities.append(tr_ppl)
        val_perplexities.append(vl_ppl)

        epoch_end_time = time.time()
        epoch_minutes, epoch_seconds = divmod(
            int(epoch_end_time - epoch_start_time), 60
        )
        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.3f} acc={tr_acc:.2%} ppl={tr_ppl:.1f} │ "
            f"val_loss={vl_loss:.3f} acc={vl_acc:.2%} ppl={vl_ppl:.1f} │ "
            f"Time: {epoch_minutes}m {epoch_seconds}s"
        )

    # ---- 6. Save ----
    MODELS_DIR = "models"
    LOGGING_DIR = "logging"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Get filename without extension for prefixing
    script_name = os.path.basename(__file__)
    filename_base = os.path.splitext(script_name)[0]

    # --- Save model (timestamped and latest) ---
    model_save_path = os.path.join(MODELS_DIR, f"{filename_base}_{ts}.pth")
    latest_model_path = os.path.join(MODELS_DIR, f"{filename_base}_latest.pth")
    save_dict = {
        "model_state": MODEL.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }
    torch.save(save_dict, model_save_path)
    torch.save(save_dict, latest_model_path)
    print(f"Model saved to {model_save_path} and {latest_model_path}")

    # --- Save loss plot (timestamped and latest) ---
    loss_plot_path = os.path.join(LOGGING_DIR, f"{filename_base}_losses_{ts}.png")
    latest_loss_plot_path = os.path.join(
        LOGGING_DIR, f"{filename_base}_losses_latest.png"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

    # Add a title to the figure
    fig.suptitle(
        f"{script_name}\n{DATA_PATH}\nEpochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}",
        fontsize=14,
    )

    epochs_range = range(1, EPOCHS + 1)
    ax1.plot(epochs_range, train_losses, label="Training Loss")
    ax1.plot(epochs_range, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, label="Training Accuracy")
    ax2.plot(epochs_range, val_accs, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs_range, train_perplexities, label="Training Perplexity")
    ax3.plot(epochs_range, val_perplexities, label="Validation Perplexity")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Perplexity")
    ax3.set_title("Training and Validation Perplexity")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(loss_plot_path)
    plt.savefig(latest_loss_plot_path)
    plt.close(fig)
    print(f"Plots saved to {loss_plot_path} and {latest_loss_plot_path}")

    # Calculate and print total runtime
    total_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")

    # ---- 7. Demo ----
    DEMO = "what is your name"
    print("\nSRC :", DEMO)
    print("PRED:", infer(MODEL, DEMO, src_vocab, tgt_vocab, inv_tgt_vocab))
