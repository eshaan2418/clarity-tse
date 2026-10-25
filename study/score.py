"""Keyword scoring of typed word-recognition responses (order-free multiset match on content words)."""

from __future__ import annotations

STOP = set("A AN THE AND OR OF TO IN ON AT IS IT HE SHE WE I YOU THEY BE WAS ARE AS BY FOR WITH THAT THIS "
           "HIS HER HIM THEM MY ME OUR BUT SO NOT NO HAD HAVE HAS DO DID".split())


def keywords(text: str) -> list[str]:
    return [w for w in normalise(text) if w not in STOP]


def normalise(text: str) -> list[str]:
    return "".join(c if c.isalpha() or c in " '" else " " for c in text.upper()).replace("'", "").split()


def score(typed: str, transcript: str) -> tuple[int, int]:
    """(# keywords correctly reported, # keywords). Multiset match, order-free (standard keyword scoring)."""
    kw = keywords(transcript)
    pool = normalise(typed)
    hit = 0
    for w in kw:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit, len(kw)
