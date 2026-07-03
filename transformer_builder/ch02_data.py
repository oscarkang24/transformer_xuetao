"""Chapter 2 — Working with text data.

Builds up the input pipeline: simple word-level tokenizers, then the
sliding-window dataset and dataloader used to produce (input, target)
pairs for next-token prediction. BPE tokenization is delegated to
tiktoken (GPT-2 encoding), as in the book.
"""

import re

import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTokenizerV1:
    """Word-level tokenizer over a fixed vocabulary (listing 2.3).

    Splits on punctuation and whitespace; raises KeyError on words that
    are not in the vocabulary.
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before punctuation
        return re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)


class SimpleTokenizerV2:
    """Word-level tokenizer with <|unk|> and <|endoftext|> handling (listing 2.4)."""

    UNK = "<|unk|>"
    END_OF_TEXT = "<|endoftext|>"

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else self.UNK for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)


def build_vocab(text, add_special_tokens=True):
    """Build a word-level vocabulary from raw text (section 2.3)."""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    tokens = sorted(set(preprocessed))
    if add_special_tokens:
        tokens.extend([SimpleTokenizerV2.END_OF_TEXT, SimpleTokenizerV2.UNK])
    return {token: i for i, token in enumerate(tokens)}


class GPTDatasetV1(Dataset):
    """Sliding-window dataset of (input, target) chunks (listing 2.5).

    The target sequence is the input shifted one position to the right,
    which is exactly the next-token-prediction objective.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """Create the GPT-2 BPE dataloader over a raw text corpus (listing 2.6)."""
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
