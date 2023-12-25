import tiktoken


def extract(d: dict, key: str) -> str:
    """
    Get the value to a key from a dict if it is not none and of length larger than 0.
    Else return None.
    """

    v = d.get(key)
    if v != None:
        if len(v) > 0:
            return v
    return None


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
