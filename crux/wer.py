#!/usr/bin/env python3
"""Word error rate for the CRUX harness (whisper.apr #89, R2).

Normalization is deliberately small and stated here, so a WER number can be
reproduced by hand: lowercase; drop every character that is not a letter,
digit, apostrophe or whitespace; split on whitespace. Levenshtein distance over
words, divided by the reference word count. An empty reference gives 0.0 when
the hypothesis is also empty, else 1.0.

Usage: wer.py <reference_file> <hypothesis_file>   -> prints the WER as a float
"""
import re
import sys


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: wer.py <reference_file> <hypothesis_file>")
    with open(sys.argv[1], encoding="utf-8") as f:
        ref = words(f.read())
    with open(sys.argv[2], encoding="utf-8") as f:
        hyp = words(f.read())
    print(f"{wer(ref, hyp):.6f}")
