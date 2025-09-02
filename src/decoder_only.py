"""Autoregressive Decoder-Only Transformer Model"""

# pylint: disable=C3001,R0914,R0913,R0917
import os
import sys
import re
import math
import datetime
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.mha import MHA  # pylint: disable=C0413

FILE_NAME = "0_nano"
# FILE_NAME = "1_mini"
# FILE_NAME = "2_full"
DATA_PATH = f"Datasets/tiny_shakespeare_{FILE_NAME}.txt"

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
    """Word-level tokenizer that ignores punctuation."""
    # Find only word characters, ignoring punctuation
    words = re.findall(r"\b\w+\b", text)
    return [word.lower() for word in words]


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict, dict]:
    """Builds vocab and inverse vocab from texts, filtering by min_freq."""
    frequency: dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
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


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    Returns sequences of token IDs for next token prediction.
    """

    def __init__(self, text: str, vocab: dict, seq_len: int):
        self.seq_len = seq_len
        tokens = tokenize(text)
        self.token_ids = encode(tokens, vocab)

        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len // 2):  # 50% overlap
            if i + seq_len < len(self.token_ids):
                self.sequences.append(
                    self.token_ids[i : i + seq_len + 1]
                )  # +1 for target

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]  # input: [0:n-1], target: [1:n]


def collate(batch, pad_id):
    """
    Pads sequences and creates causal mask for autoregressive training.
    """
    inputs, targets = zip(*batch)
    seq_len = max(len(seq) for seq in inputs)

    # Pad sequences
    input_batch = torch.full((seq_len, len(batch)), pad_id, dtype=torch.long)
    target_batch = torch.full((seq_len, len(batch)), pad_id, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_batch[: len(inp), i] = torch.tensor(inp)
        target_batch[: len(tgt), i] = torch.tensor(tgt)

    # Create attention mask (causal)
    attn_mask = torch.triu(torch.full((seq_len, seq_len), MASK_VAL), diagonal=1)
    attn_mask.fill_diagonal_(0.0)

    # Create padding mask
    key_pad_mask = (input_batch == pad_id).T  # (B, T)

    return input_batch, target_batch, attn_mask.float(), key_pad_mask


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


class CustomDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer (Pre-Norm, Decoder-Only)
    ---------------------------------------------------------
    This class implements a single layer of a decoder-only Transformer, as used in
    autoregressive language models (e.g., GPT). It is designed for educational clarity.

    Key Components:
    - Multi-Head Self-Attention (MHA): Allows each position in the sequence to attend
        to all previous positions (including itself), enabling the model to capture dependencies and context across the sequence.
    - Feedforward Network (FFN): Applies a position-wise nonlinearity to each token embedding, increasing model capacity.
    - Layer Normalization (Pre-Norm): Normalizes inputs before each sub-layer, which helps with training stability.
    - Residual Connections: Add the input of each sub-layer to its output, allowing gradients to flow more easily and improving optimization.
    - Dropout: Regularizes the model to prevent overfitting.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        nhead (int): Number of attention heads in MHA.
        dim_feedforward (int): Hidden size of the feedforward network.
        dropout (float): Dropout probability.

    Forward Pass:
        src: (Seq_Len, Batch, D_Model) - Input sequence embeddings.
        src_mask: (Seq_Len, Seq_Len) - Causal mask to prevent attending to future tokens.
        src_key_padding_mask: (Batch, Seq_Len) - Mask to ignore padding tokens.

    Returns:
        (Seq_Len, Batch, D_Model) - Output sequence embeddings after attention and FFN.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Multi-Head Self-Attention: Each token can attend to all previous tokens.
        # Uses a custom MHA implementation for educational purposes.
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

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

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for a single decoder layer.

        1. Pre-Norm & Self-Attention:
            - Normalize the input.
            - Apply multi-head self-attention, using a causal mask to ensure that each position can only attend to itself and previous positions.
            - Add the result to the input (residual connection).
        2. Pre-Norm & Feedforward:
            - Normalize the result.
            - Apply the feedforward network (linear -> activation -> dropout -> linear).
            - Add the result to the input (residual connection).

        Args:
            src: (Seq_Len, Batch, D_Model) - Input embeddings.
            src_mask: (Seq_Len, Seq_Len) - Causal mask for autoregressive modeling.
            src_key_padding_mask: (Batch, Seq_Len) - Mask for padding tokens.

        Returns:
            (Seq_Len, Batch, D_Model) - Output embeddings.
        """
        x = src

        #### 1. Self-Attention Block ####
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=src_mask,  # Prevents attending to future tokens
            key_padding_mask=src_key_padding_mask,  # Ignores padding tokens
            need_weights=False,  # We don't need attention weights for training
        )
        x = x + self.dropout1(attn_output)  # Residual connection and dropout

        #### 2. Feedforward Block ####
        norm_x_ff = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(norm_x_ff))))
        x = x + self.dropout2(ff_output)  # Residual connection and dropout

        return x


class TransformerModel(nn.Module):
    """
    Decoder-Only Transformer Model for Autoregressive Language Modeling
    -------------------------------------------------------------------
    This class implements a full decoder-only Transformer, suitable for tasks like
    next-token prediction, text generation, and language modeling (e.g., GPT-style).

    Model Architecture:
    - Embedding Layer: Maps token IDs to dense vectors.
    - Positional Encoding: Adds information about token positions (since attention is permutation-invariant).
    - Stack of Decoder Layers: Each layer consists of multi-head self-attention and a feedforward network, as described above.
    - Final Layer Normalization: Normalizes the output before projection.
    - Output Projection: Maps the final hidden states to vocabulary logits.

    Args:
        vocab (dict): Vocabulary mapping tokens to integer IDs.

    Forward Pass:
        x: (Seq_Len, Batch) - Input token IDs.
        attn_mask: (Seq_Len, Seq_Len) - Causal mask for autoregressive modeling.
        key_pad_mask: (Batch, Seq_Len) - Mask for padding tokens.

    Returns:
        (Seq_Len, Batch, Vocab_Size) - Logits for each token in the vocabulary.
    """

    def __init__(self, vocab):
        super().__init__()
        self.d_model = D_MODEL

        # Embedding layer: Converts token IDs to dense vectors.
        self.embed = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=D_MODEL, padding_idx=0
        )

        # Positional encoding: Adds position information to embeddings.
        self.position_encode = PositionalEncoding(
            d_model=D_MODEL, dropout=DROPOUT, max_len=MAX_GEN_LEN + 50
        )

        # Stack of decoder layers: Each layer is a CustomDecoderLayer.
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

        # Final normalization before output projection.
        self.final_norm = nn.LayerNorm(D_MODEL)

        # Output projection: Maps hidden states to vocabulary logits.
        self.projection = nn.Linear(in_features=D_MODEL, out_features=len(vocab))

    def forward(self, x, attn_mask, key_pad_mask):
        """
        Forward pass for the decoder-only Transformer.

        1. Embedding and Positional Encoding:
            - Convert input token IDs to embeddings.
            - Scale embeddings by sqrt(d_model) (as in Vaswani et al.).
            - Add positional encodings to provide order information.

        2. Pass through Decoder Layers:
            - Each layer applies multi-head self-attention and a feedforward network, using causal and padding masks to ensure proper autoregressive behavior.

        3. Final Normalization and Output Projection:
            - Normalize the output.
            - Project to vocabulary size to obtain logits for each token.

        Args:
            x: (Seq_Len, Batch) - Input token IDs.
            attn_mask: (Seq_Len, Seq_Len) - Causal mask for autoregressive modeling.
            key_pad_mask: (Batch, Seq_Len) - Mask for padding tokens.

        Returns:
            (Seq_Len, Batch, Vocab_Size) - Logits for each token in the vocabulary.
        """
        #### 1. Embedding and Positional Encoding ####
        # Shape: (Seq_Len, Batch, D_Model)
        x = self.position_encode(self.embed(x) * math.sqrt(self.d_model))

        #### 2. Decoder Layers ####
        # Each layer applies self-attention and feedforward transformations.
        for layer in self.decoder_layers:
            x = layer(
                src=x,
                src_mask=attn_mask.to(x.device),
                src_key_padding_mask=key_pad_mask.to(x.device),
            )

        #### 3. Final Normalization and Output Projection ####
        out = self.final_norm(x)

        return self.projection(out)  # Shape: (Seq_Len, Batch, Vocab_Size)


def train_epoch(model, loader, optimizer_, loss_criterion_, pad_id):
    """Trains the model for one epoch."""
    model.train()
    tot_loss = tot_batches = 0
    tot_tok = tot_correct = 0
    for inputs, targets, attn_mask, key_pad_mask in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        attn_mask, key_pad_mask = attn_mask.to(DEVICE), key_pad_mask.to(DEVICE)

        optimizer_.zero_grad()
        logits = model(inputs, attn_mask, key_pad_mask)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()

        # ---- accuracy (token level, ignores PAD) ----
        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = targets.ne(pad_id)
            tot_correct += (pred.eq(targets) & mask).sum().item()
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
    for inputs, targets, attn_mask, key_pad_mask in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        attn_mask, key_pad_mask = attn_mask.to(DEVICE), key_pad_mask.to(DEVICE)

        logits = model(inputs, attn_mask, key_pad_mask)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        pred = logits.argmax(-1)
        mask = targets.ne(pad_id)
        tot_correct += (pred.eq(targets) & mask).sum().item()
        tot_tok += mask.sum().item()

        tot_loss += loss.item()
        tot_batches += 1
    return tot_loss / tot_batches, tot_correct / tot_tok


def infer(model, prompt, vocab, inv_vocab, max_len=100, temperature=0.0, top_k=20):
    """Generates text using the trained model with temperature and top-k sampling.

    Args:
        model: Trained Transformer model.
        prompt (str): Input prompt text.
        vocab (dict): Token to id mapping.
        inv_vocab (dict): Id to token mapping.
        max_len (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature in [0.0, 2.0].
            - 0.0 => deterministic (greedy argmax)
            - >0  => probabilistic sampling after scaling logits by 1/temperature
        top_k (int): If >0, restrict sampling to the top-k tokens (default 20).
    """
    model.eval()
    tokens = tokenize(prompt)
    token_ids = encode(tokens, vocab)

    # Add BOS token if not present
    if not token_ids or token_ids[0] != vocab[BOS_TOKEN]:
        token_ids = [vocab[BOS_TOKEN]] + token_ids

    # Track original prompt length (excluding BOS token)
    original_prompt_length = len(token_ids) - 1  # -1 for BOS token

    # Clamp controls
    temperature = 0.0 if temperature is None else float(temperature)
    temperature = max(0.0, min(2.0, temperature))
    top_k = 0 if top_k is None else int(top_k)
    top_k = max(0, top_k)

    for _ in range(max_len):
        # Prepare input
        x = torch.tensor(token_ids, device=DEVICE).unsqueeze(1)  # (T, 1)
        seq_len = x.size(0)

        # Create masks
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), MASK_VAL, device=DEVICE), diagonal=1
        )
        attn_mask.fill_diagonal_(0.0)
        key_pad_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=DEVICE)

        # Get next token prediction
        with torch.no_grad():
            logits = model(x, attn_mask, key_pad_mask)

            last_logits = logits[-1, 0]
            if temperature == 0.0:
                # Deterministic greedy decoding
                next_token_id = int(torch.argmax(last_logits, dim=-1).item())
            else:
                # Temperature scaling and optional top-k sampling
                scaled_logits = last_logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(
                        scaled_logits, min(top_k, scaled_logits.size(-1))
                    )
                    scaled_logits[scaled_logits < values[-1]] = -float("Inf")

                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_token_id == vocab[EOS_TOKEN]:
            break

        token_ids.append(next_token_id)

    # Convert back to text - return only newly generated words, not the original prompt
    all_words = [inv_vocab.get(i, UNK_TOKEN) for i in token_ids[1:]]  # Skip BOS
    # Return only the words generated after the original prompt
    new_words = all_words[original_prompt_length:]
    return " ".join(new_words)


if __name__ == "__main__":
    start_time = time.time()
    # ---- 1. Load data ----
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text_data = f.read()
    print(f"Loaded text with {len(text_data):,} characters")

    # ---- 2. Vocab ----
    vocab, inv_vocab = build_vocab([text_data], MIN_FREQ)
    print(f"Vocab size: {len(vocab):,}")
    PAD_ID = vocab[PAD_TOKEN]

    # ---- 3. Dataset / DataLoader ----
    full_ds = TextDataset(text_data, vocab, INPUT_MAX_SEQ_LEN)
    TRAIN_SZ = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(
        full_ds,
        [TRAIN_SZ, len(full_ds) - TRAIN_SZ],
        generator=torch.Generator().manual_seed(42),
    )
    collate_func = lambda b: collate(b, pad_id=vocab[PAD_TOKEN])
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_func)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_func)

    # ---- 4. Model / Optim / Loss ----
    MODEL = TransformerModel(vocab).to(DEVICE)
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])

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
    print("\n")
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
        "vocab": vocab,
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
    DEMO = "We are accounted poor citizens, the patricians good."
    print("\nPROMPT:", DEMO)
    print(
        "GENERATED:",
        infer(MODEL, DEMO, vocab, inv_vocab, max_len=50, top_k=10, temperature=0.8),
    )
