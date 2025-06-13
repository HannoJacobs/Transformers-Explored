import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.decoder_only import *

MODEL_PATH = "models/decoder_only_latest.pth"
INFER_TEXTS = [
    "We are accounted poor citizens, the patricians good.",
    "Would you proceed especially against",
    "Against him first: he's a very",
    "Consider you what services",
    "Very well; and could be content to",
]


def load_model(ckpt_path: str):
    device = DEVICE
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    inv_vocab = {i: w for w, i in vocab.items()}
    model = TransformerModel(vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, inv_vocab


model, vocab, inv_vocab = load_model(MODEL_PATH)
print("\n--- Text Generation ---")
for text in INFER_TEXTS:
    generated = infer(model, text, vocab, inv_vocab, max_len=50)
    print(f"PROMPT    : {text}")
    print(f"GENERATED : {generated}")
    print("-" * 50)
