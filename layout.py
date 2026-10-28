"""
Reading a page as rows and columns rather than as a wall of text.

This is the change that matters most. Flattening OCR output to a string
throws away the horizontal position of every figure, and in a set of
accounts the horizontal position *is* the meaning: it tells you whether a
number belongs to this year or last year.

Run over a real balance sheet, the flat text reads:

    Debtors 6 493,712 517,202

A regex looking for a number after the word Debtors finds 6, or picks up
2025 from the heading two rows above. Keeping the coordinates instead gives:

    label "Debtors"   figures 493,712 @ x1430   517,202 @ x1962
    year headings     2025 @ x1525             2024 @ x2057

and choosing the current year becomes arithmetic rather than guesswork.
"""

import io
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

from figures import YEAR, is_money_cell, repair, to_number


class Word:
    __slots__ = ("x", "text")

    def __init__(self, x: int, text: str):
        self.x = x
        self.text = text

    def __repr__(self):
        return f"{self.text}@{self.x}"


class Cell:
    __slots__ = ("x", "token", "value")

    def __init__(self, x: int, token: str, value: float):
        self.x = x
        self.token = token
        self.value = value

    def __repr__(self):
        return f"Cell({self.token}@{self.x})"


class Row:
    __slots__ = ("words", "cells", "label", "text", "top")

    def __init__(self, words: List[Word], cells: List[Cell], top: int):
        self.words = words
        self.cells = cells
        self.top = top
        # Show the repaired figure rather than the raw token, so the text a
        # human or an LLM reads matches the value that was parsed from it.
        fixed = {c.x: c.token for c in cells}
        self.text = " ".join(fixed.get(w.x, w.text) for w in words)
        self.label = " ".join(w.text for w in words if w.x not in fixed).strip()

    def __repr__(self):
        return f"Row({self.label!r}, {self.cells})"


def page_image(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def page_rows(img: Image.Image, psm: int = 6, lang: str = "eng") -> List[Row]:
    """OCR one page and group the words back into rows, keeping x positions.

    pytesseract is imported here rather than at the top so that an install
    running only the vision model does not need Tesseract present at all.
    """
    import pytesseract
    data = pytesseract.image_to_data(
        img, lang=lang, config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT)

    grouped = defaultdict(list)
    for i, token in enumerate(data["text"]):
        if not token.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        centre = data["left"][i] + data["width"][i] // 2
        grouped[key].append((centre, data["top"][i], token))

    rows = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key])
        words = [Word(x, t) for x, _, t in ordered]
        top = min(t for _, t, _ in ordered)

        cells = []
        for word in words:
            if is_money_cell(word.text):
                value = to_number(word.text)
                if value is not None:
                    cells.append(Cell(word.x, repair(word.text), value))
        rows.append(Row(words, cells, top))
    return rows


def rows_from_text_layer(page) -> List[Row]:
    """Same row model, built from an embedded text layer instead of OCR.

    Most Companies House filings are scans, but not all of them, and a real
    text layer is both faster and perfectly accurate. Using one code path
    for both means the extraction logic never has to care which it got.
    """
    rows = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            words, top = [], None
            for span in line.get("spans", []):
                for token in span["text"].split():
                    x0, y0, x1, _ = span["bbox"]
                    words.append(Word(int((x0 + x1) / 2), token))
                    top = int(y0) if top is None else min(top, int(y0))
            if not words:
                continue
            # spans give one centre per span, so spread tokens across it
            cells = []
            for word in words:
                if is_money_cell(word.text):
                    value = to_number(word.text)
                    if value is not None:
                        cells.append(Cell(word.x, repair(word.text), value))
            rows.append(Row(words, cells, top or 0))
    rows.sort(key=lambda r: r.top)
    return rows


def page_text(rows: List[Row]) -> str:
    return "\n".join(r.text for r in rows)


def year_columns(rows: List[Row]) -> Dict[int, int]:
    """Find the year heading row and return {year: x centre}.

    Takes the highest row on the page carrying two or more distinct years,
    which in a set of accounts is the column heading and never a figure.
    """
    best = None
    for row in rows:
        positions = {}
        for word in row.words:
            fixed = repair(word.text)
            if YEAR.match(fixed):
                positions.setdefault(int(fixed), word.x)
        if len(positions) >= 2 and (best is None or row.top < best[0]):
            best = (row.top, positions)
    return best[1] if best else {}


Column = Tuple[int, int, Tuple[float, float]]

# How a nil figure is printed, once repair() has normalised the dashes.
NIL = ("-", "—", "–", "nil", "none")


def current_year_column(years: Dict[int, int]) -> Optional[Column]:
    """Return (year, x centre, (lower, upper)) for the most recent year.

    The bounds are the midpoints to the neighbouring columns, which is what
    decides whether a figure is this year or the comparative.
    """
    if len(years) < 2:
        return None
    latest = max(years)
    x = years[latest]
    ordered = sorted(years.values())

    upper = float("inf")
    for pos in ordered:
        if pos > x:
            upper = (x + pos) / 2
            break
    lower = 0.0
    for pos in reversed(ordered):
        if pos < x:
            lower = (x + pos) / 2
            break
    return latest, x, (lower, upper)


def pick_current(row: Row, column: Optional[Column]) -> Optional[Cell]:
    """Choose the figure belonging to the most recent year.

    Falls back to the leftmost figure when no year heading was found, which
    is the right default: accounts always present the current year first.
    """
    if column is None:
        return row.cells[0] if row.cells else None

    _, x, (lower, upper) = column

    # Statements with an items column and a totals column per year put the
    # year heading over the totals, so last year's items column can sit
    # inside this year's bounds. "Deferred tax 10 - (1,025,711)" is nil this
    # year and 1,025,711 last year, and position alone read it the other
    # way. When the row holds one entry per year and one is a dash, the
    # order decides: this year is always presented first.
    # A dash inside the caption ("Loans - directors") is not a nil figure,
    # so only dashes after the last word of the caption count.
    cell_x = {c.x for c in row.cells}
    caption_end = max((w.x for w in row.words
                       if any(ch.isalpha() for ch in w.text)
                       and repair(w.text) not in NIL), default=-1)
    entries = sorted(
        [(c.x, c) for c in row.cells] +
        [(w.x, None) for w in row.words
         if w.x not in cell_x and w.x > caption_end
         and repair(w.text) in NIL],
        key=lambda e: e[0])
    if len(entries) == 2 and any(c is None for _, c in entries):
        first_x, first = entries[0]
        return first if first is not None else Cell(first_x, "0", 0.0)

    inside = [c for c in row.cells if lower <= c.x < upper]
    if inside:
        return inside[0]

    # The current year is usually the leftmost column, so its lower bound
    # runs all the way to the margin and takes in the Note column. A money
    # cell can never be a note reference, but a bare word can: on
    # "Turnover 3 5 -" in thousands the 3 is the note and 5 is the figure,
    # and on "Debtors 2 3 3" the 2 is the note. So bare words are only
    # accepted within half a column spacing either side of the heading, and
    # the one nearest the heading wins rather than the leftmost.
    near = max(lower, x - (upper - x)) if upper != float("inf") else lower

    # If row.cells didn't have a figure inside the column bounds, check row.words:
    # 1. Could be a nil/dash symbol representing zero
    # 2. Could be a 1 or 2 digit number (e.g. on statements in thousands or dormant filings)
    for word in sorted(row.words, key=lambda w: abs(w.x - x)):
        if near <= word.x < upper:
            fixed = repair(word.text)
            if fixed in NIL:
                return Cell(word.x, "0", 0.0)
            val = to_number(word.text)
            if val is not None:
                return Cell(word.x, repair(word.text), val)

    return None
