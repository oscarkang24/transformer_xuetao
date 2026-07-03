"""Train the tiny "toy" GPT that powers the interactive web app.

Uses the exact GPTModel from transformer_builder (chapter 4) with a small
config, trains it on a tiny patterned corpus, and exports the weights to
docs/js/weights.js so the browser can run real inference with real trained
weights (no fetch needed — works from file://).

nn.Linear stores weights as [out, in] and computes y = x @ W.T; the export
transposes every Linear weight to [in, out] so the JS side can compute
y = x @ W directly.

Usage:
    python scripts/export_toy_model.py
"""

import json
import os
import re

import torch

from transformer_builder import GPTModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "docs", "js", "weights.js")

# A tiny patterned corpus. Deliberately repetitive so a 2-layer model
# learns it cold — the web app's probability charts then show genuinely
# peaked, meaningful distributions instead of noise.
CORPUS = """
every effort moves you forward .
every effort moves you closer .
every step moves you forward .
the cat sat on the mat .
the cat sat on the rug .
the dog sat on the rug .
the dog chased the cat .
the cat chased the mouse .
the mouse ran under the table .
the bird flew over the house .
the bird sat in the tree .
the dog ran in the park .
the cat ran under the table .
a bird flew over the tree .
a mouse ran under the house .
a dog chased a bird .
you build the model step by step .
you build the model layer by layer .
the model learns word by word .
attention moves information forward .
every layer moves information forward .
""".strip()

TOY_CONFIG = {
    "vocab_size": None,  # filled in after the vocab is built
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def tokenize(text):
    """Word-level tokenization. Must match docs/js/tokenizer.js exactly."""
    return [t for t in re.findall(r"[a-z]+|\.", text.lower())]


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    vocab.append("<|unk|>")
    return vocab


def make_batches(token_ids, context_length):
    """All sliding windows of (input, target) over the corpus, stride 1."""
    inputs, targets = [], []
    for i in range(len(token_ids) - context_length):
        inputs.append(token_ids[i : i + context_length])
        targets.append(token_ids[i + 1 : i + context_length + 1])
    return torch.tensor(inputs), torch.tensor(targets)


def round_nested(x, ndigits=5):
    if isinstance(x, list):
        return [round_nested(v, ndigits) for v in x]
    return round(x, ndigits)


def linear_weight(module):
    """nn.Linear weight as [in, out] for JS row-vector matmul."""
    return round_nested(module.weight.detach().T.tolist())


def main():
    torch.manual_seed(123)

    tokens = tokenize(CORPUS)
    vocab = build_vocab(tokens)
    stoi = {s: i for i, s in enumerate(vocab)}
    token_ids = [stoi[t] for t in tokens]

    cfg = dict(TOY_CONFIG, vocab_size=len(vocab))
    print(f"Vocab size: {len(vocab)}, corpus tokens: {len(token_ids)}")

    model = GPTModel(cfg)
    inputs, targets = make_batches(token_ids, cfg["context_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    model.train()
    for step in range(401):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    model.eval()

    blocks = []
    for block in model.trf_blocks:
        blocks.append(
            {
                "W_query": linear_weight(block.att.W_query),
                "W_key": linear_weight(block.att.W_key),
                "W_value": linear_weight(block.att.W_value),
                "out_proj_w": linear_weight(block.att.out_proj),
                "out_proj_b": round_nested(block.att.out_proj.bias.detach().tolist()),
                "ff_w1": linear_weight(block.ff.layers[0]),
                "ff_b1": round_nested(block.ff.layers[0].bias.detach().tolist()),
                "ff_w2": linear_weight(block.ff.layers[2]),
                "ff_b2": round_nested(block.ff.layers[2].bias.detach().tolist()),
                "norm1_scale": round_nested(block.norm1.scale.detach().tolist()),
                "norm1_shift": round_nested(block.norm1.shift.detach().tolist()),
                "norm2_scale": round_nested(block.norm2.scale.detach().tolist()),
                "norm2_shift": round_nested(block.norm2.shift.detach().tolist()),
            }
        )

    export = {
        "config": cfg,
        "vocab": vocab,
        "tok_emb": round_nested(model.tok_emb.weight.detach().tolist()),
        "pos_emb": round_nested(model.pos_emb.weight.detach().tolist()),
        "blocks": blocks,
        "final_norm_scale": round_nested(model.final_norm.scale.detach().tolist()),
        "final_norm_shift": round_nested(model.final_norm.shift.detach().tolist()),
        "out_head": linear_weight(model.out_head),
    }

    # Reference forward pass so the JS implementation can be verified
    # against PyTorch (see scripts/verify_js_model.mjs).
    test_prompt = ["every", "effort", "moves", "you"]
    test_ids = [stoi[t] for t in test_prompt]
    with torch.no_grad():
        test_logits = model(torch.tensor([test_ids]))[0, -1]
    export["test"] = {
        "input_ids": test_ids,
        "expected_last_logits": round_nested(test_logits.tolist()),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("// Generated by scripts/export_toy_model.py — do not edit by hand.\n")
        f.write("// Tiny GPT trained with transformer_builder (PyTorch), exported for\n")
        f.write("// in-browser inference. Linear weights are pre-transposed to [in, out].\n")
        f.write("const MODEL_WEIGHTS = ")
        f.write(json.dumps(export, separators=(",", ":")))
        f.write(";\n")
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.0f} KB)")

    # Show what the trained model predicts, as a sanity check.
    probs = torch.softmax(test_logits, dim=-1)
    top = torch.topk(probs, 5)
    print("P(next | 'every effort moves you'):")
    for p, i in zip(top.values, top.indices):
        print(f"  {vocab[i]!r}: {p.item():.3f}")


if __name__ == "__main__":
    main()
