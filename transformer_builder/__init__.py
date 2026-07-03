"""Transformer Builder — a GPT built step by step.

Follows the chapter progression of Sebastian Raschka's
*Build a Large Language Model (From Scratch)*:

- ch02_data:      working with text data (tokenization, datasets, dataloaders)
- ch03_attention: attention mechanisms (simple -> causal -> multi-head)
- ch04_gpt:       the GPT model (LayerNorm, GELU, FFN, TransformerBlock, GPTModel)
- ch05_train:     pretraining on unlabeled data (loss, training loop, decoding)
"""

from transformer_builder.ch02_data import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    GPTDatasetV1,
    create_dataloader_v1,
)
from transformer_builder.ch03_attention import (
    SelfAttentionV1,
    SelfAttentionV2,
    CausalAttention,
    MultiHeadAttention,
)
from transformer_builder.ch04_gpt import (
    GPT_CONFIG_124M,
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    generate_text_simple,
)
from transformer_builder.ch05_train import (
    calc_loss_batch,
    calc_loss_loader,
    train_model_simple,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

__all__ = [
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "GPTDatasetV1",
    "create_dataloader_v1",
    "SelfAttentionV1",
    "SelfAttentionV2",
    "CausalAttention",
    "MultiHeadAttention",
    "GPT_CONFIG_124M",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "generate_text_simple",
    "calc_loss_batch",
    "calc_loss_loader",
    "train_model_simple",
    "generate",
    "text_to_token_ids",
    "token_ids_to_text",
]
