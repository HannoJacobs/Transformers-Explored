"""Fine-tune the decoder-only model on an instruction dataset.

This script:
- Loads a pre-trained checkpoint from models/decoder_only_latest.pth (by default)
- Builds a dataset from instruct_dataset.txt using the saved vocab
- Fine-tunes the model
- Saves a timestamped and a *_latest.pth checkpoint for the finetuned model
"""

import os
import sys
import time
import math
import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

# Ensure we can import project modules when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.decoder_only import (  # pylint: disable=C0413
    TransformerModel,
    TextDataset,
    collate,
    train_epoch,
    eval_epoch,
    PAD_TOKEN,
    DEVICE,
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    INPUT_MAX_SEQ_LEN as DEFAULT_SEQ_LEN,
)


# =====================
# Global Configuration
# =====================
# Edit these variables to configure fine-tuning without CLI args
BASE_MODEL_PATH = "models/decoder_only_latest.pth"
DATASET_PATH = "Datasets/instruct_0.txt"
EPOCHS = 3
BATCH_SIZE = DEFAULT_BATCH_SIZE
SEQ_LEN = DEFAULT_SEQ_LEN
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.1
SAVE_PREFIX = "models/decoder_only_finetuned"


def load_base_checkpoint(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base model checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            "Checkpoint must be a dict containing 'model_state' and 'vocab'"
        )
    vocab = ckpt["vocab"]
    model = TransformerModel(vocab).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
    return model, vocab


def main():
    start_time = time.time()
    print(f"Using device: {DEVICE}")
    print(f"Loading base model from: {BASE_MODEL_PATH}")

    # 1) Load base model and vocab
    model, vocab = load_base_checkpoint(BASE_MODEL_PATH)
    pad_id = vocab[PAD_TOKEN]

    # 2) Load dataset text
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_text = f.read()
    print(f"Loaded dataset with {len(dataset_text):,} characters from {DATASET_PATH}")

    # 3) Build dataset and splits using existing vocab (unknown tokens map to <unk>)
    full_ds = TextDataset(dataset_text, vocab, SEQ_LEN)
    if len(full_ds) < 2:
        raise ValueError(
            "Dataset is too small after preprocessing. Provide more data or reduce seq_len."
        )

    val_size = max(1, int(VAL_SPLIT * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collate_fn = lambda b: collate(b, pad_id=pad_id)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4) Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 5) Fine-tuning loop
    epochs_range = range(1, EPOCHS + 1)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for ep in epochs_range:
        epoch_start = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_dl, optimizer, loss_criterion, pad_id
        )
        vl_loss, vl_acc = eval_epoch(model, val_dl, loss_criterion, pad_id)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        tr_ppl = math.exp(tr_loss)
        vl_ppl = math.exp(vl_loss)
        elapsed = int(time.time() - epoch_start)
        mm, ss = divmod(elapsed, 60)
        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.3f} acc={tr_acc:.2%} ppl={tr_ppl:.1f} │ "
            f"val_loss={vl_loss:.3f} acc={vl_acc:.2%} ppl={vl_ppl:.1f} │ "
            f"Time: {mm}m {ss}s"
        )

    # 6) Save finetuned model
    os.makedirs("models", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base = os.path.basename(SAVE_PREFIX)
    dir_prefix = os.path.dirname(SAVE_PREFIX) or "models"
    os.makedirs(dir_prefix, exist_ok=True)

    model_save_path = os.path.join(dir_prefix, f"{base}_{ts}.pth")
    latest_model_path = os.path.join(dir_prefix, f"{base}_latest.pth")
    save_dict = {
        "model_state": model.state_dict(),
        "vocab": vocab,
    }
    torch.save(save_dict, model_save_path)
    torch.save(save_dict, latest_model_path)
    print(f"Finetuned model saved to {model_save_path} and {latest_model_path}")

    total_seconds = int(time.time() - start_time)
    m, s = divmod(total_seconds, 60)
    print(f"Total finetune runtime: {m}m {s}s")


if __name__ == "__main__":
    main()
