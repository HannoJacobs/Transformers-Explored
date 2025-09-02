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

# Temperature settings with paired top_k values for combined effects demo
TEMP_CONFIGS = [
    (0.0, 1, "❄️"),  # Deterministic + top_1 (only best token)
    (0.5, 5, "🌤️"),  # Low creativity + small top_k
    (1.0, 20, "🌡️"),  # Balanced + medium top_k
    (2.0, 50, "🔥"),  # High creativity + large top_k
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

print("\nTemperature + Top-K Effects Demo")
print("=" * 40)

for text in INFER_TEXTS:
    print(f"\nPrompt: {text}")
    print("-" * 30)

    for temp, top_k, emoji in TEMP_CONFIGS:
        generated = infer(
            model, text, vocab, inv_vocab, max_len=20, temperature=temp, top_k=top_k
        )
        print(f"{emoji} {temp:.1f}(k={top_k}): {generated}")

    print()
