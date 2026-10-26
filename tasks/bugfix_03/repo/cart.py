"""A tiny shopping-cart helper."""


def add_item(item, items=[]):
    """Return the cart with ``item`` appended."""
    items.append(item)
    return items


def total_price(items, prices):
    return sum(prices[item] for item in items)