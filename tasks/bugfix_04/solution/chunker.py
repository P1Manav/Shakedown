"""Utilities for batching work items."""


def chunk(items, size):
    """Split ``items`` into consecutive lists of at most ``size`` elements.

    The final chunk may be shorter than ``size``.  Every input item must appear
    exactly once in the output.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(chunks):
    return [item for group in chunks for item in group]