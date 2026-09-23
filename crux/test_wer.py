"""Positive controls for crux/wer.py (whisper.apr #89, R2).

Each case must produce exactly its expected WER: a planted substitution and a
planted deletion must register, normalization alone must not, and a
timestamped line is NOT stripped (passing -nt is the caller's job).

Run: python3 -m pytest crux/test_wer.py   (or: python3 crux/test_wer.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer import wer, words  # noqa: E402

CASES = [
    ("the birds can use", "The birds can use.", 0.0),
    ("the birds can use", "the birds can lose", 0.25),
    ("the birds can use", "the birds can", 0.25),
    ("", "", 0.0),
    ("", "you", 1.0),
    ("the birds can use", "[00:00:00.000 --> 00:00:01.500]  The birds can use", 2.0),
]


def test_cases():
    for ref, hyp, want in CASES:
        assert abs(wer(words(ref), words(hyp)) - want) < 1e-9, (ref, hyp, want)


if __name__ == "__main__":
    test_cases()
    print(f"ok: {len(CASES)} cases")
