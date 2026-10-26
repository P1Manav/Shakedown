"""A tiny shopping-cart helper."""


def add_item(item, items=None):
    """Return the cart with ``item`` appended.

    A new list is created only when the caller does not supply a cart, so
    calls cannot leak state through a shared default argument.
    """
    if items is None:
        items = []
    items.append(item)
    return items


def total_price(items, prices):
    return sum(prices[item] for item in items)