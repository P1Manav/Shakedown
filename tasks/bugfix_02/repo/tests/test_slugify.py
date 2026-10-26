from slugify import slugify


def test_lowercases():
    assert slugify("Hello World") == "hello-world"


def test_collapses_punctuation_and_whitespace():
    assert slugify("  Release   Notes!!  ") == "release-notes"


def test_strips_edges():
    assert slugify("---Already Dashed---") == "already-dashed"


def test_numbers_are_kept():
    assert slugify("Python 3.11 Is Fast") == "python-3-11-is-fast"