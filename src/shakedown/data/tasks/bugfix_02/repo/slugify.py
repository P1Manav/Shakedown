"""URL slug generation."""

import re


def slugify(text):
    """Turn a title into a URL-safe slug.

    Expected: lower-case words joined by single dashes, no leading or trailing
    dash.  Example: "Hello, World!" -> "hello-world".
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")