"""Deterministic display cleanup that preserves caption words and timings."""

from dataclasses import replace
import unicodedata

from .types import Word


def normalize_caption_punctuation(words: list[Word]) -> list[Word]:
    """Use a Latin full stop for danda punctuation on Romanized words.

    Native-script text keeps its punctuation. Do not discard arbitrary Unicode:
    accents, names, and other languages must retain their original text.
    """
    caption_letters = {c for word in words for c in word.text
                       if unicodedata.category(c).startswith("L")}
    romanized = bool(caption_letters) and all(
        "LATIN" in unicodedata.name(c, "") for c in caption_letters
    )
    normalized = []
    for word in words:
        text = word.text
        if "।" in text or "॥" in text:
            letters = [c for c in text if unicodedata.category(c).startswith("L")]
            if romanized or (letters and all("LATIN" in unicodedata.name(c, "") for c in letters)):
                text = text.replace("।", ".").replace("॥", ".")
        normalized.append(replace(word, text=text) if text != word.text else word)
    return normalized
