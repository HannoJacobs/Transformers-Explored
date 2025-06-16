"""Autoregressive Decoder-Only Transformer Model"""

# pylint: disable=C3001,R0914,R0913,R0917
import os
import re
import math
import datetime
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

# FILE_NAME = "0_nano"
# FILE_NAME = "1_mini"
FILE_NAME = "2_full"
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
MASK_VAL = float("-inf")  # -1e9  # 🔑 large negative instead of -inf
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
    """Custom Transformer Decoder Layer using nn.MultiheadAttention (pre-norm)."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,  # Input shape is (L, B, E)
        )
        # Feedforward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Forward pass for the custom decoder layer (pre-norm)."""
        x = src
        # Self-attention block with pre-normalization
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,  # More efficient if weights are not needed
        )
        x = x + self.dropout1(attn_output)

        # Feedforward block with pre-normalization
        norm_x_ff = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(norm_x_ff))))
        x = x + self.dropout2(ff_output)
        return x


class TransformerModel(nn.Module):
    """A decoder-only Transformer model for autoregressive language modeling."""

    def __init__(self, vocab):
        super().__init__()
        self.d_model = D_MODEL
        self.embed = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=D_MODEL, padding_idx=0
        )
        self.position_encode = PositionalEncoding(
            d_model=D_MODEL, dropout=DROPOUT, max_len=MAX_GEN_LEN + 50
        )

        # Create a stack of custom decoder layers
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
        self.final_norm = nn.LayerNorm(D_MODEL)  # Final normalization
        self.projection = nn.Linear(in_features=D_MODEL, out_features=len(vocab))

    def forward(self, x, attn_mask, key_pad_mask):
        """Forward pass for the decoder-only model."""
        # Embed and add positional encoding
        x = self.position_encode(self.embed(x) * math.sqrt(self.d_model))

        # Pass through the stack of custom decoder layers
        for layer in self.decoder_layers:
            x = layer(
                src=x,
                src_mask=attn_mask.to(x.device),
                src_key_padding_mask=key_pad_mask.to(x.device),
            )
        # Apply final normalization before projection
        out = self.final_norm(x)
        return self.projection(out)  # (T,B,V)


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


def infer(model, prompt, vocab, inv_vocab, max_len=100, temperature=0.8, top_k=20):
    """Generates text using the trained model with temperature and top-k sampling."""
    model.eval()
    tokens = tokenize(prompt)
    token_ids = encode(tokens, vocab)

    # Add BOS token if not present
    if not token_ids or token_ids[0] != vocab[BOS_TOKEN]:
        token_ids = [vocab[BOS_TOKEN]] + token_ids

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

            # Sample from the distribution
            logits = logits[-1, 0] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == vocab[EOS_TOKEN]:
            break

        token_ids.append(next_token_id)

    # Convert back to text
    words = [inv_vocab.get(i, UNK_TOKEN) for i in token_ids[1:]]  # Skip BOS
    return " ".join(words)


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
