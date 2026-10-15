"""
Turning an OCR token into a number you can trust.

Companies House re renders filed accounts as bitonal TIFF at 200 dpi. At that
resolution Tesseract makes a small, very predictable set of mistakes on
figures, and every one of them is silent: you get a number back, it is just
the wrong number.

Observed on a real filing (SC284050):

    312.365    should be 312,365     comma read as a full stop
    3,261.113  should be 3,261,113   same
    $0,045     should be 50,045      leading 5 read as $
    L00        should be 100         1 read as L
    {469,694   should be (469,694    bracket misread

The repairs below only ever run on tokens that already look like a figure,
so ordinary prose is never touched.
"""

import os
import re
from typing import Optional

# Characters Tesseract swaps for digits on low resolution bitonal scans.
CONFUSABLE = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "L": "1", "i": "1",
    "Z": "2",
    "S": "5", "$": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
    "{": "(", "}": ")", "[": "(", "]": ")",
    "\u00a3": "", "\u20ac": "", "\u00a5": "",
    "\u2014": "-", "\u2013": "-", "\u2019": "", "\u2018": "",
    " ": "",
}

# The repairs above are aimed at Tesseract on a 200 dpi bitonal scan. A
# vision model reads characters in context and almost never swaps an L for a
# 1, so FIGURE_REPAIR=light drops the letter substitutions and keeps only
# the things that are still true of any engine: currency symbols, bracket
# shapes, and the full stop that should have been a thousands separator.
LIGHT = {
    "{": "(", "}": ")", "[": "(", "]": ")",
    "\u00a3": "", "\u20ac": "", "\u00a5": "",
    "\u2014": "-", "\u2013": "-", "\u2019": "", "\u2018": "",
    " ": "",
}

REPAIR_MODE = os.getenv("FIGURE_REPAIR", "full").lower().strip()


def set_repair_mode(mode: str) -> None:
    """full for Tesseract, light for a vision model."""
    global REPAIR_MODE
    REPAIR_MODE = mode.lower().strip()


# What a figure in a set of accounts is allowed to look like once repaired.
MONEY_TOKEN = re.compile(r"^\(?-?\d{1,3}(?:,\d{3})+\)?$|^\(?-?\d+\)?$")

# A four digit number on its own is a column heading, not a figure.
YEAR = re.compile(r"^(19|20)\d{2}$")


def looks_numeric(token: str) -> bool:
    """At least half the characters are digits. Keeps the repairs away from
    words like 'Debtors' that happen to contain a confusable letter."""
    if not token:
        return False
    digits = sum(c.isdigit() for c in token)
    return digits >= 1 and digits * 2 >= len(token.strip("()-.,"))


def repair(token: str) -> str:
    """Apply the confusable map, then fix thousands separators.

    UK statutory accounts are presented in whole pounds, so pence are not
    shown. A full stop followed by exactly three digits is therefore always
    a misread comma, never a decimal point. That single rule fixes the worst
    error in the set, because reading 312.365 as three hundred pounds rather
    than three hundred thousand is a factor of a thousand.
    """
    if not looks_numeric(token):
        return token

    table = LIGHT if REPAIR_MODE == "light" else CONFUSABLE
    out = "".join(table.get(c, c) for c in token)
    # trailing junk that is not part of a figure
    out = out.strip(".:;'\"")
    # full stop acting as a thousands separator
    out = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", ",", out)
    return out


def looks_like_year(fixed: str) -> bool:
    """A year, whatever punctuation is stuck to it.

    The plain pattern is anchored, so it catches 2024 but not (2024, which
    is how a comparative written as "(2024 30)" ended up being read as minus
    two thousand and twenty four. Strip the brackets before deciding.

    The thousands separator is the tell that distinguishes the two cases: a
    genuine figure of two thousand and twenty four pounds is written 2,024
    in a set of accounts, so a four digit run with no separator sitting in
    the year range is a year.
    """
    if "," in fixed:
        return False
    return bool(YEAR.match(fixed.strip("()-")))


def to_number(token: str) -> Optional[float]:
    """Parse a repaired token into a float. Brackets mean negative.

    Returns None for anything that is not a figure, including bare years,
    which is what stops a 2025 column heading being read as a value.
    """
    fixed = repair(token)
    if looks_like_year(fixed):
        return None
    if not MONEY_TOKEN.match(fixed):
        return None

    negative = fixed.startswith("(") or fixed.startswith("-")
    digits = fixed.strip("()-").replace(",", "")
    if not digits.isdigit():
        return None
    value = float(digits)
    return -value if negative else value


def is_money_cell(token: str) -> bool:
    """A figure rather than a note reference or a year.

    Note reference columns hold 1 or 2 digit numbers. Requiring either a
    thousands separator or four digits keeps them out without needing to
    know where the note column sits.
    """
    fixed = repair(token)
    if looks_like_year(fixed):
        return False
    if not MONEY_TOKEN.match(fixed):
        return False
    digits = fixed.strip("()-").replace(",", "")
    return "," in fixed or len(digits) >= 4


def scale_for(text: str) -> tuple:
    """Accounts are often presented in thousands or millions. Detect the
    heading so 4,320 under a £'000 column is read as 4.32 million."""
    if re.search(r"\u00a3\s?'?m\b|\bin millions\b|\bmillions of\b", text, re.I):
        return 1_000_000, "millions"
    if re.search(r"\u00a3\s?'?000|\bin thousands\b|\bthousands of\b", text, re.I):
        return 1000, "thousands"
    return 1, "units"
