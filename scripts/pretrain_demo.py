"""End-to-end pretraining demo (chapter 5).

Trains a small GPT on "The Verdict" (the short story used throughout the
book) and prints generated samples as the loss drops. Downloads the text
on first run; pass --epochs to change training length.

Usage:
    python scripts/pretrain_demo.py [--epochs 10]
"""

import argparse
import os
import urllib.request

import tiktoken
import torch

from transformer_builder import (
    GPTModel,
    create_dataloader_v1,
    train_model_simple,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "the-verdict.txt")

GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,   # shortened from 1024 to keep CPU training fast
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def load_corpus():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Downloading corpus to {DATA_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text_data = load_corpus()
    split_idx = int(0.9 * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=2,
        max_length=GPT_CONFIG_SMALL["context_length"],
        stride=GPT_CONFIG_SMALL["context_length"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=2,
        max_length=GPT_CONFIG_SMALL["context_length"],
        stride=GPT_CONFIG_SMALL["context_length"],
        shuffle=False,
        drop_last=False,
    )

    model = GPTModel(GPT_CONFIG_SMALL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    print("\n--- Sampling with temperature=1.4, top_k=25 ---")
    model.eval()
    token_ids = generate(
        model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=30,
        context_size=GPT_CONFIG_SMALL["context_length"],
        temperature=1.4,
        top_k=25,
    )
    print(token_ids_to_text(token_ids, tokenizer))

    ckpt_path = os.path.join(os.path.dirname(DATA_PATH), "model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
