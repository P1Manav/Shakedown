from slugify import slugify


def test_slugify_hidden_special_characters():
    assert slugify("Special @#$% Characters & Spaces") == "special-characters-spaces"
    assert slugify("---Leading and Trailing---") == "leading-and-trailing"
    assert slugify("Multiple----Hyphens___And   Tabs") == "multiple-hyphens-and-tabs"
