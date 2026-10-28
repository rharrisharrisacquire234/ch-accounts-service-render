"""
Does the balance sheet add up?

Every figure this service returns is read off one page, and OCR mistakes
are silent: an 8 read as a 3 still gives a number. A balance sheet is the
one place in a set of accounts where that can be caught without a human,
because its lines are bound together by arithmetic:

    sum of fixed asset items                 = fixed assets
    sum of current asset items               = current assets
    current assets - creditors < 1 year      = net current assets
    fixed assets + net current assets        = total assets less current
                                               liabilities
    that, less creditors > 1 year,
      provisions and the like                = net assets
    capital and reserves                     = net assets

Each check that has the lines it needs is run on the page net assets was
read from, in the page's own units, and the result says which ones held.
A failure does not say which figure is wrong, only that one of the lines on
that page was misread or missed, so the row wants a look before it is
trusted.
"""

import re
from typing import Any, Dict, List, Optional

import fields
from layout import current_year_column, pick_current, year_columns

# Rounding to the nearest pound, or thousand, can leave a subtotal one out.
TOLERANCE = 2

CAPITAL_AND_RESERVES = re.compile(r"^capital\s+and\s+reserves\b", re.I)
NET_CURRENT_LINE = re.compile(r"^net\s+current\b", re.I)
# Lines between total assets less current liabilities and net assets that
# are deductions even when printed without brackets.
DEDUCTION = re.compile(
    r"creditor|provision|deferred|accrual|liabilit|loan|lease|borrowing", re.I)

CAPTIONS = (fields.TALCL_LABEL, fields.CREDITORS_WITHIN_LABEL,
            fields.NET_ASSETS_LABEL, NET_CURRENT_LINE,
            re.compile(r"due\s+after\s+(?:more\s+than\s+)?one\s+year", re.I))


class Line:
    __slots__ = ("label", "value", "text")

    def __init__(self, label: str, value: Optional[float], text: str):
        self.label = label
        self.value = value
        self.text = text


def _clean(label: str) -> str:
    """The caption without note references, brackets and stray marks."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z/'’ ]+", " ", label)).strip()


def _lines(rows, column) -> List[Line]:
    """One entry per balance sheet line, wrapped captions rejoined.

    "Creditors: amounts falling due within one" / "year 8 (5,814,001)" is a
    single line on the page, printed over two rows."""
    raw = []
    for row in rows:
        cell = pick_current(row, column)
        raw.append(Line(_clean(row.label), cell.value if cell else None,
                        row.text))
    out, skip = [], False
    for index, line in enumerate(raw):
        if skip:
            skip = False
            continue
        following = raw[index + 1] if index + 1 < len(raw) else None
        if (line.label and line.value is None and following is not None
                and following.value is not None and following.label):
            joined = f"{line.label} {following.label}"
            continues = following.label[0].islower()
            completes = any(p.search(joined) and not p.search(line.label)
                            for p in CAPTIONS)
            if continues or completes:
                out.append(Line(joined, following.value,
                                f"{line.text} / {following.text}"))
                skip = True
                continue
        out.append(line)
    return out


def _find(lines: List[Line], pattern, exclude=None, start: int = 0) -> int:
    for index in range(start, len(lines)):
        label = lines[index].label
        if label and pattern.search(label) and not (
                exclude and exclude.search(label)):
            return index
    return -1


def _section(lines: List[Line], head: int):
    """(items, total) for the section headed at lines[head]."""
    if lines[head].value is not None:
        return [], lines[head].value
    items = []
    for line in lines[head + 1:head + 14]:
        if line.label and fields.SECTION_END.match(line.label):
            # "Total current assets 8,080" closes its own section, but
            # "Total assets less current liabilities" belongs to the next.
            if (line.value is not None and re.match(r"^total\b", line.label, re.I)
                    and not fields.TALCL_LABEL.search(line.label)):
                return items, line.value
            if (line.value is not None
                    and fields.EQUITY_LABEL.search(line.label)):
                return items, line.value
            break
        if line.value is None:
            continue
        if not line.label:
            return items, line.value
        items.append(line.value)
    if len(items) == 1:
        return items, items[0]
    return items, None


def _signed_net(line: Line) -> float:
    value = line.value
    label = line.label.lower()
    if (("liabilit" in label and "assets" not in label)
            or "deficit" in label):
        return -abs(value)
    return value


def _fmt(value: float) -> str:
    return f"({abs(value):,.0f})" if value < 0 else f"{value:,.0f}"


def balance_sheet(pages: List[Dict[str, Any]],
                  found: Dict[str, Any]) -> Dict[str, Any]:
    net = found.get("net_assets")
    if not net:
        return {"status": "not checked", "passed": 0, "run": 0,
                "checks": [], "summary": "no net assets line to check from"}
    page = next((p for p in pages if p["index"] == net.get("page")), None)
    if page is None:
        return {"status": "not checked", "passed": 0, "run": 0,
                "checks": [], "summary": "net assets page not in hand"}

    rows = page["rows"]
    column = current_year_column(year_columns(rows))
    mult, _ = fields.scale_for(page["text"])
    lines = _lines(rows, column)
    checks: List[Dict[str, Any]] = []

    def check(name: str, got: float, want: float, working: str):
        ok = abs(got - want) <= TOLERANCE
        checks.append({"name": name, "ok": ok,
                       "working": f"{working} = {_fmt(got)}, "
                                  f"page shows {_fmt(want)}"})

    # ---- the lines everything else hangs off
    i_na = _find(lines, fields.NET_ASSETS_LABEL, exclude=fields.NET_CURRENT)
    if i_na < 0 or lines[i_na].value is None:
        i_na = _find(lines, fields.EQUITY_LABEL)
    net_value = _signed_net(lines[i_na]) if i_na >= 0 and \
        lines[i_na].value is not None else None

    i_fa = _find(lines, fields.FIXED_ASSETS_LABEL)
    i_ca = _find(lines, fields.CURRENT_ASSETS_LABEL)
    i_cw = _find(lines, fields.CREDITORS_WITHIN_LABEL,
                 exclude=fields.CREDITORS_EXCLUDE)
    i_nca = _find(lines, NET_CURRENT_LINE)
    i_tl = _find(lines, fields.TALCL_LABEL)

    fixed_items, fixed = _section(lines, i_fa) if i_fa >= 0 else ([], None)
    current_items, current = _section(lines, i_ca) if i_ca >= 0 else ([], None)
    within = abs(lines[i_cw].value) if i_cw >= 0 and \
        lines[i_cw].value is not None else None
    nca = None
    if i_nca >= 0 and lines[i_nca].value is not None:
        nca = lines[i_nca].value
        label = lines[i_nca].label.lower()
        if "liabilit" in label and "assets" not in label:
            nca = -abs(nca)
    talcl = lines[i_tl].value if i_tl >= 0 else None

    # ---- the identities
    if fixed is not None and len(fixed_items) >= 2:
        check("fixed asset items add up", sum(fixed_items), fixed,
              " + ".join(_fmt(v) for v in fixed_items))
    if current is not None and len(current_items) >= 2:
        check("current asset items add up", sum(current_items), current,
              " + ".join(_fmt(v) for v in current_items))
    if current is not None and within is not None and nca is not None:
        check("net current assets", current - within, nca,
              f"current assets {_fmt(current)} - creditors within one year "
              f"{_fmt(within)}")
    if talcl is not None:
        if nca is not None:
            base, base_name = nca, "net current assets"
        elif current is not None and within is not None:
            base, base_name = current - within, "current assets less creditors"
        else:
            base = None
        if base is not None and (fixed is not None or i_fa < 0):
            check("total assets less current liabilities",
                  (fixed or 0) + base, talcl,
                  f"fixed assets {_fmt(fixed or 0)} + {base_name} {_fmt(base)}")

    if talcl is not None and net_value is not None and i_tl < i_na:
        total, parts = talcl, [f"{_fmt(talcl)}"]
        for line in lines[i_tl + 1:i_na]:
            if line.value is None or not line.label:
                continue
            if line.value < 0 or DEDUCTION.search(line.label):
                total -= abs(line.value)
                parts.append(f"- {abs(line.value):,.0f}")
            else:
                total += line.value
                parts.append(f"+ {line.value:,.0f}")
        check("net assets from total assets less current liabilities",
              total, net_value, " ".join(parts))

    i_cr = _find(lines, CAPITAL_AND_RESERVES)
    if i_cr >= 0 and net_value is not None:
        items, equity = _section(lines, i_cr)
        if equity is not None:
            if len(items) >= 2:
                check("capital and reserves items add up", sum(items), equity,
                      " + ".join(_fmt(v) for v in items))
            check("capital and reserves equals net assets", equity,
                  net_value, "capital and reserves")

    # ---- the figures this service reports, against the page
    debtors = found.get("debtors")
    if (debtors and current is not None and debtors.get("page") == page["index"]
            and debtors["value"] > current * mult + TOLERANCE * mult):
        checks.append({"name": "debtors within current assets", "ok": False,
                       "working": f"debtors {debtors['value']:,.0f} exceed "
                                  f"current assets {current * mult:,.0f}"})
    assets = found.get("total_assets")
    if (assets and assets.get("page") == page["index"]
            and fixed is not None and current is not None):
        check("reported total assets", assets["value"] / mult,
              fixed + current,
              f"reported total assets in page units")

    passed = sum(1 for c in checks if c["ok"])
    if not checks:
        status, summary = "not checked", "balance sheet lines not found"
    elif passed == len(checks):
        status, summary = "pass", f"{passed}/{len(checks)} checks hold"
    else:
        status = "fail"
        summary = "; ".join(f"{c['name']}: {c['working']}"
                            for c in checks if not c["ok"])
    return {"status": status, "passed": passed, "run": len(checks),
            "page": page["index"], "checks": checks, "summary": summary[:400]}
