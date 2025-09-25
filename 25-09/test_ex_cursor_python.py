import os
import sys

# Ensure the test can import the module when run from the repo root
sys.path.append(os.path.dirname(__file__))

from ex_cursor_python import slugify
import pytest
from ex_cursor_python import search_internet as search_internet_fn
import ex_cursor_python as ex_mod


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_mixed_case():
    assert slugify("PyTest Rocks") == "pytest-rocks"


def test_slugify_hyphens_preserved():
    assert slugify("Already-Slug") == "already-slug"


def test_slugify_multiple_spaces():
    # Current implementation replaces each space with a hyphen and does not strip
    assert slugify("  a  b ") == "--a--b-" 


def test_slugify_empty_string():
    assert slugify("") == ""


def test_slugify_leading_trailing_spaces():
    # Current implementation lowercases and replaces spaces with hyphens without trimming
    assert slugify("  Hello  ") == "--hello--"


def test_slugify_consecutive_spaces():
    assert slugify("a   b") == "a---b"


def test_slugify_tabs_and_newlines():
    # Tabs/newlines are not treated as spaces by current implementation
    assert slugify("a\tb\nc") == "a\tb\nc"


def test_slugify_unicode_accents():
    # Accented characters are lowercased but not normalized
    assert slugify("Città È Bella") == "città-è-bella"


def test_slugify_punctuation():
    # Punctuation is preserved
    assert slugify("hello, world!") == "hello,-world!"


def test_slugify_numbers_and_underscores():
    assert slugify("123_abc DEF") == "123_abc-def"


def test_slugify_none_input():
    # Current implementation raises AttributeError when passed None
    with pytest.raises(AttributeError):
        slugify(None)  # type: ignore[arg-type] 


def test_search_internet_success(monkeypatch):
    class DummyDdgs:
        def __init__(self):
            self.received = None
        def ddg(self, q):
            self.received = q
            return {"results": ["a", "b"]}
    dummy = DummyDdgs()
    monkeypatch.setattr(ex_mod, "ddgs", dummy, raising=False)

    out = search_internet_fn("test query")
    assert out == {"results": ["a", "b"]}
    assert dummy.received == "test query"


def test_search_internet_empty_query(monkeypatch):
    class DummyDdgs:
        def ddg(self, q):
            return []
    monkeypatch.setattr(ex_mod, "ddgs", DummyDdgs(), raising=False)

    out = search_internet_fn("")
    assert out == []


def test_search_internet_propagates_errors(monkeypatch):
    class FailingDdgs:
        def ddg(self, q):
            raise RuntimeError("ddgs error")
    monkeypatch.setattr(ex_mod, "ddgs", FailingDdgs(), raising=False)

    with pytest.raises(RuntimeError):
        search_internet_fn("anything") 

