"""Train the tiny *diffusion* language model for the web app's stage 7.

Same architecture, corpus, and size as the autoregressive toy model
(export_toy_model.py) with two changes that turn it into an
absorbing-state discrete diffusion model (a la MaskGIT / LLaDA):

1. The causal mask is removed — every position attends to every other,
   because the model predicts masked tokens from *both* sides.
2. Training is mask-and-reconstruct instead of next-token: each step a
   random fraction of positions is replaced by <|mask|> and the model is
   trained to recover the originals at exactly those positions.

Generation (implemented in docs/js/stages.js) then starts from an
all-<|mask|> sequence and unmasks the most confident positions over a
few parallel refinement rounds — the discrete analogue of iterative
denoising.

Usage:
    python scripts/export_diffusion_model.py
"""

import json
import os
import sys

import torch

from transformer_builder import GPTModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_toy_model import (  # noqa: E402
    CORPUS, TOY_CONFIG, tokenize, build_vocab, round_nested, linear_weight,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "docs", "js", "weights_diffusion.js")

MASK_TOKEN = "<|mask|>"


def make_windows(token_ids, length):
    return torch.tensor(
        [token_ids[i : i + length] for i in range(len(token_ids) - length)]
    )


def main():
    torch.manual_seed(321)

    tokens = tokenize(CORPUS)
    vocab = build_vocab(tokens)          # words + <|unk|>, ids identical to the AR model
    vocab.append(MASK_TOKEN)             # one extra id, used only by this model
    stoi = {s: i for i, s in enumerate(vocab)}
    mask_id = stoi[MASK_TOKEN]
    token_ids = [stoi[t] for t in tokens]

    cfg = dict(TOY_CONFIG, vocab_size=len(vocab))
    print(f"Vocab size: {len(vocab)} (mask id {mask_id}), corpus tokens: {len(token_ids)}")

    model = GPTModel(cfg)
    # Remove causality: the mask buffer is upper-triangular ones; zeroing it
    # makes masked_fill_ a no-op, so attention becomes bidirectional.
    for block in model.trf_blocks:
        block.att.mask.zero_()

    windows = make_windows(token_ids, cfg["context_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    model.train()
    for step in range(801):
        optimizer.zero_grad()
        batch = windows[torch.randint(0, len(windows), (32,))]
        # Random corruption level per sample, then Bernoulli masking —
        # the discrete counterpart of "add a random amount of noise".
        t = torch.rand(batch.shape[0], 1).clamp(min=0.15)
        masked = torch.rand(batch.shape) < t
        masked[masked.sum(dim=1) == 0, 0] = True  # never a fully clean sample
        corrupted = torch.where(masked, torch.full_like(batch, mask_id), batch)

        logits = model(corrupted)
        loss = torch.nn.functional.cross_entropy(logits[masked], batch[masked])
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
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
        "config": dict(cfg, causal=False),
        "vocab": vocab,
        "mask_id": mask_id,
        "tok_emb": round_nested(model.tok_emb.weight.detach().tolist()),
        "pos_emb": round_nested(model.pos_emb.weight.detach().tolist()),
        "blocks": blocks,
        "final_norm_scale": round_nested(model.final_norm.scale.detach().tolist()),
        "final_norm_shift": round_nested(model.final_norm.shift.detach().tolist()),
        "out_head": linear_weight(model.out_head),
    }

    # Reference forward pass on a half-masked input so the JS bidirectional
    # path can be verified against PyTorch.
    test_ids = [stoi["the"], mask_id, stoi["sat"], stoi["on"], mask_id, stoi["mat"]]
    with torch.no_grad():
        test_logits = model(torch.tensor([test_ids]))[0]
    export["test"] = {
        "input_ids": test_ids,
        "expected_logits_pos1": round_nested(test_logits[1].tolist()),
        "expected_logits_pos4": round_nested(test_logits[4].tolist()),
    }

    with open(OUT_PATH, "w") as f:
        f.write("// Generated by scripts/export_diffusion_model.py — do not edit by hand.\n")
        f.write("// Bidirectional (non-causal) twin of the toy GPT, trained as an\n")
        f.write("// absorbing-state discrete diffusion model (mask-and-reconstruct).\n")
        f.write("const DIFFUSION_WEIGHTS = ")
        f.write(json.dumps(export, separators=(",", ":")))
        f.write(";\n")
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")

    # Sanity check: what does it infill for "the <mask> sat on <mask> mat"?
    for pos in (1, 4):
        probs = torch.softmax(test_logits[pos], dim=-1)
        top = torch.topk(probs, 3)
        guesses = ", ".join(f"{vocab[i]!r} {p.item():.2f}" for p, i in zip(top.values, top.indices))
        print(f"P(position {pos} | 'the ? sat on ? mat'): {guesses}")


if __name__ == "__main__":
    main()
