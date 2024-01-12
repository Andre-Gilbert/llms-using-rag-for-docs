"""Utility functions."""
from itertools import islice
from typing import Any, Generator, Iterable

import tiktoken


def _batched(iterable: Iterable, n: int):
    """Batches data into tuples of length n (the last batch may be shorter)."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text: str, chunk_size: int) -> Generator[tuple, Any, None]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks_iterator = _batched(tokens, chunk_size)
    yield from chunks_iterator


def get_text_from_tokens(tokens) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = encoding.decode(tokens)
    return text
