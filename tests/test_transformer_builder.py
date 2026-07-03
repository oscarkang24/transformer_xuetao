"""Sanity tests for every stage of the build, chapter by chapter."""

import pytest
import torch

from transformer_builder import (
    SimpleTokenizerV2,
    create_dataloader_v1,
    SelfAttentionV1,
    SelfAttentionV2,
    CausalAttention,
    MultiHeadAttention,
    LayerNorm,
    FeedForward,
    TransformerBlock,
    GPTModel,
    generate_text_simple,
    generate,
    calc_loss_batch,
)
from transformer_builder.ch02_data import build_vocab

TEST_CFG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 48,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


# --- Chapter 2 -----------------------------------------------------------


def test_simple_tokenizer_roundtrip():
    text = "Hello, world. Is this-- a test?"
    vocab = build_vocab(text)
    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode(text)
    # "--" is split into its own token and rejoined with spaces, exactly
    # like the book's decode (it only strips spaces before punctuation).
    assert tokenizer.decode(ids) == "Hello, world. Is this -- a test?"


def test_simple_tokenizer_unknown_words():
    vocab = build_vocab("Hello, world.")
    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode("Hello, unseen.")
    assert tokenizer.decode(ids) == "Hello, <|unk|>."


def test_dataloader_targets_are_inputs_shifted():
    text = "In the beginning was the word and the word was with meaning " * 20
    try:
        loader = create_dataloader_v1(
            text, batch_size=2, max_length=8, stride=8, shuffle=False
        )
    except Exception as exc:  # tiktoken downloads the GPT-2 vocab on first use
        pytest.skip(f"GPT-2 BPE vocab unavailable offline: {type(exc).__name__}")
    inputs, targets = next(iter(loader))
    assert inputs.shape == (2, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


# --- Chapter 3 -----------------------------------------------------------


def test_self_attention_shapes():
    x = torch.rand(6, 3)
    assert SelfAttentionV1(3, 2)(x).shape == (6, 2)
    assert SelfAttentionV2(3, 2)(x).shape == (6, 2)


def test_causal_attention_ignores_future_tokens():
    torch.manual_seed(123)
    attn = CausalAttention(d_in=4, d_out=4, context_length=8, dropout=0.0)
    attn.eval()
    x = torch.rand(1, 6, 4)
    out_full = attn(x)

    # Changing a future token must not change earlier positions' outputs.
    x_perturbed = x.clone()
    x_perturbed[0, 5] = torch.rand(4)
    out_perturbed = attn(x_perturbed)
    assert torch.allclose(out_full[0, :5], out_perturbed[0, :5], atol=1e-6)


def test_multi_head_attention_shapes_and_causality():
    torch.manual_seed(123)
    mha = MultiHeadAttention(
        d_in=8, d_out=8, context_length=16, dropout=0.0, num_heads=2
    )
    mha.eval()
    x = torch.rand(2, 10, 8)
    out = mha(x)
    assert out.shape == (2, 10, 8)

    x_perturbed = x.clone()
    x_perturbed[:, -1] = torch.rand(2, 8)
    out_perturbed = mha(x_perturbed)
    assert torch.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-6)


def test_multi_head_requires_divisible_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_in=8, d_out=9, context_length=4, dropout=0.0, num_heads=2)


# --- Chapter 4 -----------------------------------------------------------


def test_layernorm_normalizes():
    ln = LayerNorm(emb_dim=16)
    x = torch.rand(2, 5, 16) * 10 + 3
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_feedforward_and_block_preserve_shape():
    x = torch.rand(2, 7, TEST_CFG["emb_dim"])
    assert FeedForward(TEST_CFG)(x).shape == x.shape
    block = TransformerBlock(TEST_CFG)
    block.eval()
    assert block(x).shape == x.shape


def test_gpt_model_output_shape():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG)
    model.eval()
    idx = torch.randint(0, TEST_CFG["vocab_size"], (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, TEST_CFG["vocab_size"])


def test_generate_text_simple_extends_sequence():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG)
    model.eval()
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 4))
    out = generate_text_simple(
        model, idx, max_new_tokens=5, context_size=TEST_CFG["context_length"]
    )
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], idx)


# --- Chapter 5 -----------------------------------------------------------


def test_loss_is_finite_and_backpropagates():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG)
    inputs = torch.randint(0, TEST_CFG["vocab_size"], (2, 8))
    targets = torch.randint(0, TEST_CFG["vocab_size"], (2, 8))
    loss = calc_loss_batch(inputs, targets, model, device="cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tok_emb.weight.grad is not None


def test_generate_topk_and_temperature():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG)
    model.eval()
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 4))
    out = generate(
        model,
        idx,
        max_new_tokens=6,
        context_size=TEST_CFG["context_length"],
        temperature=1.4,
        top_k=25,
    )
    assert out.shape == (1, 10)


def test_greedy_generate_matches_simple():
    torch.manual_seed(123)
    model = GPTModel(TEST_CFG)
    model.eval()
    idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 4))
    a = generate_text_simple(model, idx, 5, TEST_CFG["context_length"])
    b = generate(model, idx, 5, TEST_CFG["context_length"], temperature=0.0)
    assert torch.equal(a, b)
