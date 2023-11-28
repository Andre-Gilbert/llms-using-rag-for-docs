

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
