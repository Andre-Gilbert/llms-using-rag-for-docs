"""Utility functions."""
from itertools import islice
from typing import Any, Generator, Iterable

import tiktoken


def num_tokens_from_messages(messages: list[dict]) -> int:
    """Counts the number of tokens in the conversation history."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def batched(iterable: Iterable, n: int):
    """Batches data into tuples of length n (the last batch may be shorter)."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text: str, chunk_size: int) -> Generator[tuple, Any, None]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_size)
    yield from chunks_iterator


def get_text_from_tokens(tokens) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = encoding.decode(tokens)
    return text
