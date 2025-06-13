import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.seq_to_seq import *

MODEL_PATH = "models/seq_to_seq_latest.pth"
INFER_TEXTS = [
    "what is a sentence",
    "what is your name",
    "tell me a story",
    "do you know math",
    "how much money do you have",
]


def load_model(ckpt_path: str):
    device = DEVICE
    ckpt = torch.load(ckpt_path, map_location=device)
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    inv_tgt = {i: w for w, i in tgt_vocab.items()}
    model = TransformerModel(src_vocab, tgt_vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, inv_tgt


model, src_vocab, tgt_vocab, inv_tgt = load_model(MODEL_PATH)
print("\n--- Translations ---")
for text in INFER_TEXTS:
    translation = infer(model, text, src_vocab, tgt_vocab, inv_tgt)
    print(f"SRC : {text}")
    print(f"TGT : {translation}")
    print("-" * 20)
