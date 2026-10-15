"""
Turning a vision model's markdown into the same row model Tesseract gives us.

This is the piece that makes swapping the OCR engine safe. Everything
downstream of the reader (fields.py, pipeline.py, the estimator, the tests)
works on Row objects that carry an x position per token, because in a set of
accounts the horizontal position is what tells you whether a figure is this
year or the comparative.

Unlimited-OCR does not return word boxes. It returns a parsed document:
markdown headings, paragraphs, and pipe tables. So instead of throwing that
structure away we mine it. A table row like

    | Debtors | 6 | 493,712 | 517,202 |

already tells us which column each figure sits in, which is exactly what the
Tesseract path spends coordinate maths working out. We hand each cell a
synthetic x of column index times COL_STEP and the existing column picker
carries on unchanged.

Plain prose lines get an x per character, so a space aligned line still
sorts left to right sensibly, and the headcount sentence in the notes is
read the same way it always was.

If the model ever emits a table whose header row has a different number of
cells to its body rows, the column picker finds nothing inside the bounds
and falls back to the leftmost figure. That is the correct default anyway:
UK accounts always present the current year first.
"""

import html
import re
from typing import List, Tuple

from figures import is_money_cell, repair, to_number
from layout import Cell, Row, Word

# Synthetic geometry. The two constants only have to be roughly comparable
# so that a prose line and a table row on the same page do not produce wildly
# different scales for year_columns() to compare.
COL_STEP = 1000
CHAR_STEP = 100
# Table columns are numbered from the right, so RIGHT_ANCHOR is where the
# last column sits and earlier columns count backwards from it. See
# _row_from_columns for why.
RIGHT_ANCHOR = 20000

# Unlimited-OCR marks each block with its category and bounding box.
# <|det|>text [x1,y1,x2,y2]<|/det|>actual content
DET_LINE = re.compile(r"^<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)$",
                      re.DOTALL)
DET_ANY = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
REF_ANY = re.compile(r"<\|ref\|>|<\|/ref\|>|<\|grounding\|>", re.DOTALL)

# Markdown table plumbing we do not want to read as content.
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.DOTALL)
HTML_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.DOTALL)
TAG = re.compile(r"<[^>]+>")

# Markdown emphasis and heading marks. Left in place they turn "**Debtors**"
# into a label the field patterns never match.
EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
BULLET = re.compile(r"^\s{0,4}[\*\+\u2022]\s+")


def strip_markers(raw: str) -> str:
    """Remove the detection markers, drop image blocks, keep the content."""
    out = []
    for line in raw.splitlines():
        line = REF_ANY.sub("", line).rstrip()
        match = DET_LINE.match(line.strip())
        if match:
            category, content = match.group(1).strip(), match.group(2).strip()
            if category.lower() == "image":
                continue
            if content:
                out.append(content)
            continue
        line = DET_ANY.sub("", line)
        out.append(line)
    return "\n".join(out)


def _clean_cell(text: str) -> str:
    text = TAG.sub(" ", text)
    text = EMPHASIS.sub("", text)
    # The model emits real HTML entities, and SHAREHOLDERS&#x27; FUNDS does
    # not match a pattern looking for an apostrophe.
    text = html.unescape(text)
    return " ".join(text.split())


def _split_pipes(line: str) -> List[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [_clean_cell(part) for part in body.split("|")]


def _row_from_columns(columns: List[str], top: int) -> Row:
    """One table row, with its columns numbered from the right.

    Counting from the left looks obvious and is wrong. The model does not
    emit a fixed number of cells per row: on a real balance sheet it
    returned five cells for the fixed assets line and five for debtors, but
    with the empty spacer in a different place, so the same year's figures
    landed in different columns and the year heading matched only one of
    them.

    Counting from the right fixes it, because a financial table is right
    aligned by construction. The last column is the earliest comparative,
    the one before it is the current year, and that holds however many
    label or note columns the model decided to emit on the left. Trailing
    empty cells are dropped first, since they are padding to the widest row
    rather than real columns.
    """
    while columns and not columns[-1]:
        columns = columns[:-1]

    words: List[Word] = []
    cells: List[Cell] = []
    total = len(columns)
    for index, column in enumerate(columns):
        if not column:
            continue
        x = RIGHT_ANCHOR - (total - 1 - index) * COL_STEP
        tokens = column.split()
        # A cell holding one token is the normal case in a financial table.
        # A cell holding several is usually a label, so nudge the tokens
        # apart to keep reading order without inventing a second column.
        for offset, token in enumerate(tokens):
            words.append(Word(x + offset, token))
            if is_money_cell(token):
                value = to_number(token)
                if value is not None:
                    cells.append(Cell(x + offset, repair(token), value))
    return Row(words, cells, top)


def _row_from_text(line: str, top: int) -> Row:
    """A prose line. x comes from the character offset, which preserves any
    space alignment the model chose to keep."""
    words: List[Word] = []
    cells: List[Cell] = []
    for match in re.finditer(r"\S+", line):
        token = _clean_cell(match.group(0))
        if not token:
            continue
        x = match.start() * CHAR_STEP
        words.append(Word(x, token))
        if is_money_cell(token):
            value = to_number(token)
            if value is not None:
                cells.append(Cell(x, repair(token), value))
    return Row(words, cells, top)


def _html_rows(block: str) -> List[List[str]]:
    out = []
    for row in HTML_ROW.findall(block):
        columns = [_clean_cell(cell) for cell in HTML_CELL.findall(row)]
        if columns:
            out.append(columns)
    return out


def rows_from_markdown(raw: str) -> List[Row]:
    """The whole conversion. Order is preserved, one Row per source line."""
    text = strip_markers(raw)

    # HTML tables first, because a single <table> can span many lines and
    # has to be pulled out before the line by line pass sees it.
    html_blocks: List[Tuple[int, List[List[str]]]] = []

    def _stash(match):
        html_blocks.append((len(html_blocks), _html_rows(match.group(0))))
        return f"\n\u0000HTMLTABLE{len(html_blocks) - 1}\u0000\n"

    text = re.sub(r"<table[^>]*>.*?</table>", _stash, text,
                  flags=re.I | re.DOTALL)

    rows: List[Row] = []
    top = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        marker = re.match(r"^\u0000HTMLTABLE(\d+)\u0000$", stripped)
        if marker:
            for columns in html_blocks[int(marker.group(1))][1]:
                rows.append(_row_from_columns(columns, top))
                top += 1
            continue

        if TABLE_RULE.match(stripped):
            continue

        if stripped.count("|") >= 2 and stripped.startswith("|"):
            rows.append(_row_from_columns(_split_pipes(stripped), top))
            top += 1
            continue

        stripped = HEADING.sub("", stripped)
        stripped = BULLET.sub("", stripped)
        if not stripped.strip():
            continue
        rows.append(_row_from_text(line.rstrip(), top))
        top += 1

    return rows


def markdown_text(raw: str) -> str:
    """The document as flat text, for the prose patterns and for --show."""
    return "\n".join(row.text for row in rows_from_markdown(raw))
