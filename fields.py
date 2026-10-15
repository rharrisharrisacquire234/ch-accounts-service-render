"""
Pulling turnover, debtors and employee numbers out of the row model.

Three different shapes of problem, so three different approaches:

turnover   a row in the profit and loss account. Very often absent: a small
           company filing filleted accounts under section 444 does not
           deliver a P&L to the registrar at all, so the figure is simply
           not on the public record. Knowing that up front is worth more
           than any amount of OCR effort.

debtors    a row on the balance sheet, with a fuller breakdown in the notes.
           Take the balance sheet total in preference, because the note
           splits it into trade, other, and amounts due after one year.

employees  written as prose in the notes rather than tabulated:
           "Average number of employees, including directors, during the
           year was: 32 (2024: 30)". The comparative in brackets is the
           trap here, and so is the year inside it.
"""

import re
from typing import Any, Dict, List, Optional

from figures import looks_like_year, repair, scale_for


def _scaled(value: float, mult: int) -> float:
    """Apply the statement's scale to a figure, without second guessing it.

    The previous version only multiplied when the figure was already under a
    million, on the theory that a big number must already be in units. That
    reasoning is applied to the wrong thing. Scale is a property of the
    statement, not of the individual number, so the guard produced filings
    where debtors were scaled and turnover was not.

    It cost real money. On a company filing in thousands, debtors of 503,022
    became 503,022,000 while turnover of 1,934,034 stayed as it was, giving a
    company with half a billion of debtors an apparent turnover of under two
    million. The ratio is the thing every downstream estimate leans on, so an
    inconsistent pair is worse than a pair that is wrong in the same
    direction: at least then the ratio survives.
    """
    return value * mult
from layout import Row, current_year_column, pick_current, year_columns

# OCR mangles these words in predictable ways at low resolution. "Turnover"
# comes out as "Tumover" because rn reads as m. Patterns stay loose because
# a label that is missed is a figure that never gets found.
TURNOVER_LABEL = re.compile(
    r"^(?:t[uv][rm]n?[o0]ver|tum[o0]ver|reven[uv]e|"
    r"sales|gross\s+income)\b", re.I)

DEBTORS_BS_LABEL = re.compile(r"^debt[o0]rs?\b|^trade\s+and\s+other\s+receivab",
                              re.I)
TRADE_DEBTORS_LABEL = re.compile(
    r"^trade\s+debt[o0]rs?\b|^trade\s+receivab|"
    r"^amounts?\s+receivable\s+for\s+(?:goods|services|sales)", re.I)

EMPLOYEE_PROSE = re.compile(
    r"average\s+(?:monthly\s+)?number\s+of\s+"
    r"(?:emp[l1i][o0]yees?|persons?\s+emp[l1i][o0]yed|staff|personnel)"
    r"[^.\n]{0,140}", re.I)
EMPLOYEE_LABEL = re.compile(
    r"average\s+(?:monthly\s+)?number\s+of\s+"
    r"(?:emp[l1i][o0]yee|person|staff|personnel)", re.I)
# The row that carries the answer when the note is a table rather than a
# sentence. An unlabelled row of numbers under the categories is the total.
EMPLOYEE_TOTAL = re.compile(r"^total\b", re.I)
# A new numbered note ends the block, so the search does not wander into
# the next note and pick up a figure from it.
NOTE_HEADING = re.compile(r"^\d{1,2}[.)]\s")
HEADCOUNT_MAX = 500_000

# Wording that means the profit and loss account was deliberately withheld.
# The bracketed subsection in "section 444(1)" broke the previous version of
# this pattern on a real filing, so subsections are optional throughout.
NO_PROFIT_AND_LOSS = re.compile(
    r"section\s+444\s*\(?\d*\)?|"
    r"opted\s+not\s+to\s+deliver.{0,80}profit\s+and\s+loss|"
    r"profit\s+and\s+loss\s+account\s+(?:has\s+)?not\s+(?:been\s+)?"
    r"(?:delivered|filed|included)|"
    r"advantage\s+of\s+the\s+(?:small\s+compan|micro.?entit)|"
    r"(?:provisions\s+applicable\s+to\s+companies\s+)?subject\s+to\s+the\s+"
    r"(?:small\s+companies|micro.?entity)\s+regime|"
    r"prepared\s+in\s+accordance\s+with\s+the\s+"
    r"(?:micro.?entity|small\s+companies)\s+(?:provisions|regime)|"
    r"\bfilleted\b|\babridged\b",
    re.I,
)

# Filing descriptions where turnover cannot be on the record.
NO_TURNOVER_TYPES = {
    "dormant", "micro-entity", "small", "filleted", "abbreviated",
    "total-exemption-small", "total-exemption-full", "null", "unaudited",
}
TURNOVER_TYPES = {
    "full", "group", "medium", "interim", "initial",
    "audit-exemption-subsidiary",
}


def accounts_type(description: str) -> str:
    """accounts-with-accounts-type-total-exemption-full -> total-exemption-full"""
    if not description:
        return "unknown"
    marker = "accounts-type-"
    if marker in description:
        return description.split(marker, 1)[1].strip()
    return "unknown"


def turnover_on_record(atype: str) -> bool:
    if atype in NO_TURNOVER_TYPES:
        return False
    if atype in TURNOVER_TYPES:
        return True
    return True    # unknown, assume it might be there


def _label_matches(row: Row, pattern) -> bool:
    label = row.label.strip(" .:")
    return bool(label) and bool(pattern.search(label))


def find_turnover(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A Turnover or Revenue row carrying at least one figure.

    Rows with no figures are skipped, which is what keeps the accounting
    policy note out of the result. On a filleted filing the phrase
    "2.2 Turnover: Turnover is measured at the fair value of the
    consideration received" appears three times with no numbers anywhere
    near it, and the previous version was matching against exactly that.
    """
    for page in pages:
        rows = page["rows"]
        column = current_year_column(year_columns(rows))
        mult, unit = scale_for(page["text"])
        for row in rows:
            if not _label_matches(row, TURNOVER_LABEL):
                continue
            cell = pick_current(row, column)
            if cell is None or abs(cell.value) < 100:
                continue
            value = _scaled(cell.value, mult)
            return {
                "value": value,
                "raw": cell.value,
                "units": unit,
                "page": page["index"],
                "label": row.label,
                "evidence": row.text[:160],
                "year": column[0] if column else None,
            }
    return None


def find_debtors(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prefer the balance sheet total over the breakdown in the notes."""
    fallback = None
    for page in pages:
        rows = page["rows"]
        column = current_year_column(year_columns(rows))
        mult, unit = scale_for(page["text"])
        on_balance_sheet = bool(
            re.search(r"balance\s+sheet|statement\s+of\s+financial\s+position",
                      page["text"], re.I))
        for row in rows:
            if not _label_matches(row, DEBTORS_BS_LABEL):
                continue
            cell = pick_current(row, column)
            if cell is None or abs(cell.value) < 100:
                continue
            value = _scaled(cell.value, mult)
            hit = {
                "value": value,
                "raw": cell.value,
                "units": unit,
                "page": page["index"],
                "label": row.label,
                "evidence": row.text[:160],
                "year": column[0] if column else None,
                "source": "balance sheet" if on_balance_sheet else "notes",
            }
            if on_balance_sheet:
                return hit
            if fallback is None:
                fallback = hit
    return fallback


def find_employees(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Headcount is written as a sentence, not tabulated.

    Two traps in the usual wording. The comparative sits in brackets after
    the figure, and it carries a year with it, so a naive "first number
    after the label" reads 2024 rather than 32.
    """
    for page in pages:
        for match in EMPLOYEE_PROSE.finditer(page["text"]):
            value = _first_headcount(match.group(0))
            if value is not None:
                return {
                    "value": value,
                    "page": page["index"],
                    "evidence": match.group(0)[:160],
                }
        # Tabulated form, used by every larger company. The wording runs
        # "the average monthly number of employees was as follows:" and then
        # breaks into a table of categories, so the sentence itself carries
        # no number and the prose pass above finds nothing.
        hit = _employees_from_table(page)
        if hit:
            return hit
    return None


def _trailing_numbers(row) -> List[tuple]:
    """The run of numbers at the end of a row, as (x, value).

    Taking the trailing run rather than every number on the line is what
    keeps note references out. A staff costs row reads "Production 22 245
    231" where 22 is the note, and the note sits before the label ends
    rather than after the figures, so walking back from the right stops
    before it.
    """
    out = []
    for word in reversed(row.words):
        fixed = repair(word.text)
        if looks_like_year(fixed):
            break
        digits = fixed.replace(",", "")
        if not digits.isdigit():
            break
        value = int(digits)
        if not 0 < value <= HEADCOUNT_MAX:
            break
        out.append((word.x, value))
    return list(reversed(out))


def _category_label(row) -> str:
    """The row's label with the trailing headcounts stripped off.

    Row.label is built from the words that are not money cells, and a
    headcount is never a money cell, so the numbers end up inside the label.
    That matters here because an unlabelled row of numbers under the
    categories is the total, and without this it reads as another category
    and gets added to the very figures it was summarising.
    """
    tokens = row.label.split()
    while tokens and re.fullmatch(r"[\d,()\u00a3-]+", repair(tokens[-1])):
        tokens.pop()
    return " ".join(tokens).strip()


def _headcount_in_row(row, column) -> Optional[int]:
    """The current year figure from a tabulated headcount row."""
    numbers = _trailing_numbers(row)
    if not numbers:
        return None
    if column:
        # column is (year, x centre, (lower bound, upper bound)), the same
        # shape pick_current uses for money cells.
        low, high = column[2]
        for x, value in numbers:
            if low <= x < high:
                return value
    # No usable year heading. UK accounts present the current year first,
    # so the leftmost of the trailing run is the one wanted.
    return numbers[0][1]


def _employees_from_table(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the headcount out of a staff numbers table.

    This cannot go through row.cells the way every other field does.
    A cell has to look like money, which means a comma or four digits, and
    that rule is what keeps note reference columns out of the figures. A
    headcount of 32 or 245 never clears it, so on a tabulated note the row
    has no cells at all and the old code had nothing to look at. It only
    ever worked on companies employing more than a thousand people.
    """
    rows = page["rows"]
    column = current_year_column(year_columns(rows))
    for index, row in enumerate(rows):
        if not EMPLOYEE_LABEL.search(row.label):
            continue

        # Sometimes the figure is on the label row itself.
        value = _headcount_in_row(row, column)
        if value:
            return {"value": value, "page": page["index"],
                    "evidence": row.text[:160]}

        # Otherwise it is in the rows below, as categories and a total.
        categories = []
        for follower in rows[index + 1:index + 14]:
            label = _category_label(follower)
            if NOTE_HEADING.match(follower.label.strip()):
                break
            value = _headcount_in_row(follower, column)
            if value is None:
                continue
            # An unlabelled row of numbers, or one that says total, is the
            # total. Trust it over adding the categories up ourselves.
            if not label or EMPLOYEE_TOTAL.match(label):
                return {"value": value, "page": page["index"],
                        "evidence": (row.label + " / " +
                                     follower.text)[:160]}
            categories.append((follower, value))

        if categories:
            total = sum(value for _, value in categories)
            if 0 < total <= HEADCOUNT_MAX:
                return {"value": total, "page": page["index"],
                        "evidence": (row.label + " / " + " + ".join(
                            f"{f.label.strip()} {v}"
                            for f, v in categories))[:160]}
    return None


def _first_headcount(sentence: str) -> Optional[int]:
    """Take the figure before the bracketed comparative, skipping years."""
    head = sentence.split("(")[0]
    for candidate in _candidates(head) or _candidates(sentence):
        return candidate
    return None


def _candidates(text: str) -> List[int]:
    """Headcounts run into the thousands at larger groups, so the separator
    has to be allowed or 2,006 employees is read as 2."""
    out = []
    pattern = r"(?<![A-Za-z0-9.,])[\dIlOoSB]{1,3}(?:,\d{3})+(?![A-Za-z0-9])" \
              r"|(?<![A-Za-z0-9.,])[\dIlOoSB]{1,6}(?![A-Za-z0-9,])"
    for token in re.findall(pattern, text):
        separated = "," in token
        fixed = repair(token).replace(",", "")
        if not fixed.isdigit():
            continue
        n = int(fixed)
        # A bare 2024 is a year. A separated 2,006 is two thousand employees.
        if not separated and len(fixed) == 4 and 1900 <= n <= 2100:
            continue
        if 0 < n < 500_000:
            out.append(n)
    return out


def find_trade_debtors(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Trade debtors specifically, from the debtors note.

    This is the line that came from selling something. The balance sheet
    Debtors total also carries other debtors, director loans and
    intercompany balances, none of which relate to turnover, so estimating
    off the total overstates it.
    """
    for page in pages:
        rows = page["rows"]
        column = current_year_column(year_columns(rows))
        mult, unit = scale_for(page["text"])
        for row in rows:
            if not _label_matches(row, TRADE_DEBTORS_LABEL):
                continue
            cell = pick_current(row, column)
            if cell is None or abs(cell.value) < 100:
                continue
            value = _scaled(cell.value, mult)
            return {
                "value": value,
                "raw": cell.value,
                "units": unit,
                "page": page["index"],
                "label": row.label,
                "evidence": row.text[:160],
                "year": column[0] if column else None,
            }
    return None


def no_profit_and_loss(text: str) -> bool:
    return bool(NO_PROFIT_AND_LOSS.search(text))

# ---------------------------------------------------------------- balance sheet
#
# Net assets, total assets and total liabilities. Only net assets is printed
# as a single labelled line on almost every UK balance sheet. Total assets
# and total liabilities usually are not, so they are worked out from lines
# that are always there, using the identities of the statutory format:
#
#   total assets      = total assets less current liabilities
#                       + creditors falling due within one year
#   total liabilities = total assets - net assets
#
# Both workings are returned as evidence so a figure can be checked against
# the page rather than trusted blind.

NET_ASSETS_LABEL = re.compile(r"^net\s+(?:assets|liabilit)", re.I)
NET_CURRENT = re.compile(r"\bcurrent\b", re.I)
EQUITY_LABEL = re.compile(
    r"^(?:total\s+)?(?:shareholders?\W{0,2}s?\s+funds|"
    r"shareholders?\W{0,2}s?\s+(?:deficit|equity)|equity|total\s+equity)\b", re.I)
TALCL_LABEL = re.compile(r"^total\s+assets\s+less\s+current\s+liabilit", re.I)
CREDITORS_WITHIN_LABEL = re.compile(
    r"^creditors\W{0,3}\s*(?:amounts?\s+)?(?:falling\s+)?due\s+within\s+one\s+year",
    re.I)
TOTAL_ASSETS_LABEL = re.compile(r"^total\s+assets\b(?!\s+less)", re.I)
TOTAL_LIABILITIES_LABEL = re.compile(r"^total\s+liabilities\b", re.I)
CURRENT_LIABILITIES_LABEL = re.compile(r"^(?:total\s+)?current\s+liabilities\b",
                                       re.I)
BALANCE_SHEET_PAGE = re.compile(
    r"balance\s+sheet|statement\s+of\s+financial\s+position", re.I)


def _balance_row(pages: List[Dict[str, Any]], pattern,
                 exclude=None) -> Optional[Dict[str, Any]]:
    """First matching row with a figure, balance sheet pages first."""
    ordered = sorted(pages, key=lambda p: 0 if BALANCE_SHEET_PAGE.search(
        p["text"]) else 1)
    for page in ordered:
        rows = page["rows"]
        column = current_year_column(year_columns(rows))
        mult, unit = scale_for(page["text"])
        for row in rows:
            label = row.label.strip(" .:")
            if not label or not pattern.search(label):
                continue
            if exclude is not None and exclude.search(label):
                continue
            cell = pick_current(row, column)
            if cell is None:
                continue
            return {
                "value": _scaled(cell.value, mult),
                "raw": cell.value,
                "units": unit,
                "page": page["index"],
                "label": row.label,
                "evidence": row.text[:160],
                "year": column[0] if column else None,
                "source": "balance sheet" if BALANCE_SHEET_PAGE.search(
                    page["text"]) else "notes",
            }
    return None


def find_net_assets(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """NET ASSETS, or failing that the shareholders' funds line, which is the
    same number by construction. "Net current assets" is a different line
    and is excluded. A line headed "Net liabilities" is a deficit."""
    hit = _balance_row(pages, NET_ASSETS_LABEL, exclude=NET_CURRENT)
    if hit is None:
        hit = _balance_row(pages, EQUITY_LABEL)
    if hit is None:
        return None
    label = hit["label"].lower()
    if ("liabilit" in label and "assets" not in label) or "deficit" in label:
        hit["value"] = -abs(hit["value"])
    return hit


def find_total_assets(pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    explicit = _balance_row(pages, TOTAL_ASSETS_LABEL)
    if explicit:
        return explicit
    talcl = _balance_row(pages, TALCL_LABEL)
    within = _balance_row(pages, CREDITORS_WITHIN_LABEL) or \
        _balance_row(pages, CURRENT_LIABILITIES_LABEL)
    if talcl is None or within is None:
        return None
    value = talcl["value"] + abs(within["value"])
    return {
        "value": value,
        "page": talcl["page"],
        "units": talcl["units"],
        "evidence": (f"{talcl['evidence']} + {within['evidence']}")[:300],
        "working": f"total assets less current liabilities {talcl['value']:,.0f}"
                   f" + creditors within one year {abs(within['value']):,.0f}"
                   f" = {value:,.0f}",
    }


def find_total_liabilities(pages: List[Dict[str, Any]],
                           total_assets: Optional[Dict[str, Any]],
                           net_assets: Optional[Dict[str, Any]]
                           ) -> Optional[Dict[str, Any]]:
    explicit = _balance_row(pages, TOTAL_LIABILITIES_LABEL)
    if explicit:
        explicit["value"] = abs(explicit["value"])
        return explicit
    if total_assets is None or net_assets is None:
        return None
    value = total_assets["value"] - net_assets["value"]
    if value < 0:
        return None          # the two figures disagree, one is misread
    return {
        "value": value,
        "page": net_assets["page"],
        "units": net_assets.get("units"),
        "evidence": None,
        "working": f"total assets {total_assets['value']:,.0f} - net assets "
                   f"{net_assets['value']:,.0f} = {value:,.0f}",
    }


INTERCOMPANY_LABEL = re.compile(
    r"^amounts?\s+owed\s+by\s+(?:group\s+undertakings|related\s+part|"
    r"subsidiar|parent)", re.I)


def find_intercompany_debtors(pages: List[Dict[str, Any]]
                              ) -> Optional[Dict[str, Any]]:
    """Debtors owed by other group companies, from the debtors note.

    The label often wraps onto a second line ("...in which the company / has
    a participating interest 56,169 39,737"), so the figure is looked for on
    the label row first and then on the row below it."""
    for page in pages:
        rows = page["rows"]
        column = current_year_column(year_columns(rows))
        mult, _ = scale_for(page["text"])
        for index, row in enumerate(rows):
            if not INTERCOMPANY_LABEL.search(row.label.strip(" .:")):
                continue
            cell = pick_current(row, column)
            if cell is None and index + 1 < len(rows):
                cell = pick_current(rows[index + 1], column)
            if cell is not None:
                return {"value": _scaled(cell.value, mult),
                        "page": page["index"], "evidence": row.text[:160]}
    return None
