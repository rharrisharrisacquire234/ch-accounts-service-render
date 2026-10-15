"""
Run this before and after any change to the extraction rules.

    python -m pytest test_extract.py -v
    python test_extract.py            (no pytest needed)

The sample cases below are taken from real OCR output, not invented. Each
one is a mistake the previous version actually made on a real filing.
"""

import os
import sys

import figures
from figures import is_money_cell, to_number
import estimate
import fields
import markdown_rows


SAMPLE_PDF = os.getenv(
    "SAMPLE_PDF",
    os.path.join(os.path.dirname(__file__), "samples",
                 "SC284050_aa_2026-01-27.pdf"))

# What a correct read of that filing looks like. It is a filleted small
# company filing, so turnover is genuinely not in the document.
EXPECTED = {"turnover": None, "debtors": 493712.0, "employee_size": 32}


def check(name, got, want):
    ok = got == want
    print(f"  {'pass' if ok else 'FAIL'}  {name:<46} got={got!r}")
    if not ok:
        print(f"        {'':<46} want={want!r}")
    return ok


def test_separator_repair():
    """A full stop with exactly three digits after it is a misread comma.
    Reading 312.365 as three hundred pounds is a factor of a thousand out."""
    figures.set_repair_mode("full")     # these cases are Tesseract misreads
    cases = [
        ("312.365", 312365.0),      # trade debtors, real OCR output
        ("3,261.113", 3261113.0),   # fixed asset cost total
        ("95.704", 95704.0),        # trade creditors
        ("493,712", 493712.0),
        ("(404,256", -404256.0),
        ("{469,694", -469694.0),    # bracket misread
        ("$0,045", 50045.0),        # leading 5 read as a dollar sign
        ("L00", 100.0),             # 1 read as L
        ("2025", None),             # a year is not a figure
        ("6", 6.0),                 # a note reference parses...
        ("Debtors", None),
    ]
    passed = 0
    for token, want in cases:
        passed += check(f"to_number({token!r})", to_number(token), want)
    return passed, len(cases)


def test_light_repair_mode():
    """With a vision model the letter to digit repairs are not wanted. It
    reads characters in context, so an L really is an L, and a token like
    L00 is a footnote marker rather than a hundred pounds. The separator
    repair stays on either way, because UK accounts never show pence."""
    figures.set_repair_mode("light")
    cases = [("312.365", 312365.0), ("493,712", 493712.0),
             ("(404,256", -404256.0), ("L00", None), ("$0,045", None)]
    passed = 0
    for token, want in cases:
        passed += check(f"light to_number({token!r})", to_number(token), want)
    figures.set_repair_mode("full")     # leave the rest of the run as found
    return passed, len(cases)


def test_markdown_rows():
    """The vision model returns a parsed document, not word boxes. This is
    the conversion that lets every rule above carry over unchanged.

    The table below is the shape Unlimited-OCR returns for a balance sheet.
    The old failure was reading Debtors as 6, the note reference, or as
    2025 from the column heading. Both have to stay fixed on this path."""
    page = (
        "<|det|>title [12,20,600,60]<|/det|># Balance sheet\n"
        "\n"
        "| | Note | 2025 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Debtors | 6 | 493,712 | 517,202 |\n"
        "| Trade debtors | | 312,365 | 405,118 |\n"
        "| Creditors | 7 | (404,256) | (388,001) |\n"
        "\n"
        "Average number of employees, including directors, during the "
        "year was: 32 (2024 30)\n"
    )
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 3, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]

    passed = 0
    passed += check("markdown det markers stripped",
                    "<|det|>" in pages[0]["text"], False)
    debtors = fields.find_debtors(pages)
    passed += check("markdown debtors", debtors["value"] if debtors else None,
                    493712.0)
    passed += check("markdown reads the current year column",
                    debtors["year"] if debtors else None, 2025)
    trade = fields.find_trade_debtors(pages)
    passed += check("markdown trade debtors",
                    trade["value"] if trade else None, 312365.0)
    staff = fields.find_employees(pages)
    passed += check("markdown headcount",
                    staff["value"] if staff else None, 32)
    return passed, 5


def test_markdown_without_year_header():
    """If the model drops the year heading, the column picker has nothing to
    work with and falls back to the leftmost figure. That is the right
    default: UK accounts always present the current year first."""
    page = ("| Debtors | 6 | 493,712 | 517,202 |\n"
            "| Turnover | | 1,240,900 | 1,118,004 |\n")
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 0, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]
    debtors = fields.find_debtors(pages)
    turnover = fields.find_turnover(pages)
    passed = 0
    passed += check("no year header, debtors falls back left",
                    debtors["value"] if debtors else None, 493712.0)
    passed += check("no year header, turnover falls back left",
                    turnover["value"] if turnover else None, 1240900.0)
    passed += check("note reference 6 is not read as a figure",
                    debtors["value"] != 6 if debtors else False, True)
    return passed, 3


def test_employees_tabulated():
    """Larger companies tabulate the headcount instead of writing it out.

    The sentence ends in "was as follows:" and the numbers arrive underneath
    as categories, so the prose pattern matches text with no figure in it.
    The old fallback then looked at row.cells, but a cell has to look like
    money, meaning a comma or four digits, and that rule is exactly what
    keeps note reference columns out. A headcount of 245 never clears it, so
    the row had no cells and nothing was found. It only worked on companies
    employing over a thousand people, which is why every PDF filing in a
    real batch came back with no headcount.

    Three shapes have to work: an explicit total row, an unlabelled total
    row, and categories with no total at all."""
    def run(text, label):
        rows = markdown_rows.rows_from_markdown(text)
        pages = [{"index": 9, "rows": rows,
                  "text": markdown_rows.markdown_text(text)}]
        hit = fields.find_employees(pages)
        return hit["value"] if hit else None

    explicit = (
        "8. Staff Costs\n"
        "| The average monthly number of employees was as follows: | 2025 | 2024 |\n"
        "| Production | 245 | 231 |\n"
        "| Administration | 52 | 48 |\n"
        "| Total | 297 | 279 |\n")
    unlabelled = (
        "| Average number of employees, including directors | 2025 | 2024 |\n"
        "| Production | 245 | 231 |\n"
        "| Administration | 52 | 48 |\n"
        "| | 297 | 279 |\n")
    no_total = (
        "| Average number of persons employed | 2025 | 2024 |\n"
        "| Office and management | 12 | 11 |\n"
        "| Warehouse | 20 | 18 |\n")
    stops = (
        "| Average number of employees | 2025 | 2024 |\n"
        "| Production | 245 | 231 |\n"
        "9. Directors remuneration\n"
        "| Aggregate emoluments | 480,000 | 455,000 |\n")

    passed = 0
    passed += check("tabulated headcount, total row", run(explicit, ""), 297)
    passed += check("tabulated headcount, unlabelled total",
                    run(unlabelled, ""), 297)
    passed += check("tabulated headcount, no total row so summed",
                    run(no_total, ""), 32)
    passed += check("block stops at the next note", run(stops, ""), 245)

    # The prose form must still win where it exists.
    prose = ("3. Average Number of Employees\n"
             "Average number of employees, including directors, during the "
             "year was: 32 (2024: 30)\n")
    passed += check("prose headcount still preferred", run(prose, ""), 32)
    return passed, 5


def test_scale_applied_consistently():
    """A page in thousands must scale every figure on it, whatever its size.

    This is the bug that turned a company with half a billion of debtors
    into one with under two million of turnover. Debtors of 503,022 were
    under a million so they were multiplied up, and turnover of 1,934,034
    was over a million so it was left alone, on the theory that a big number
    must already be in units. Scale belongs to the statement, not to the
    number.

    The ratio is the point. Every downstream estimate divides one of these
    by the other, so a pair scaled differently is worse than a pair scaled
    wrongly together."""
    page = (
        "Consolidated Balance Sheet\n"
        "Amounts are stated in \u00a3'000\n"
        "| | Note | 2025 | 2024 |\n"
        "| Turnover | 3 | 1,934,034 | 1,801,220 |\n"
        "| Debtors | 6 | 503,022 | 488,100 |\n")
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 2, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]

    turnover = fields.find_turnover(pages)
    debtors = fields.find_debtors(pages)
    passed = 0
    passed += check("thousands page, turnover scaled",
                    turnover["value"] if turnover else None, 1_934_034_000.0)
    passed += check("thousands page, debtors scaled",
                    debtors["value"] if debtors else None, 503_022_000.0)
    passed += check("both report the same units",
                    turnover["units"] == debtors["units"] == "thousands"
                    if turnover and debtors else False, True)
    ratio = round(turnover["value"] / debtors["value"], 1) \
        if turnover and debtors else None
    passed += check("ratio survives the scaling", ratio, 3.8)
    passed += check("raw figure kept for audit",
                    turnover["raw"] if turnover else None, 1_934_034.0)
    return passed, 5


def test_bracketed_year_is_not_a_figure():
    """A comparative written as "(2024 30)" in the employees note was being
    read as minus two thousand and twenty four, because the year pattern is
    anchored and the opening bracket made it miss.

    The separator is what tells the two apart. A real figure of two thousand
    and twenty four pounds is written 2,024 in a set of accounts, so a four
    digit run with no separator in the year range is a year."""
    figures.set_repair_mode("full")
    cases = [("(2024", None), ("2024", None), ("(1990", None),
             ("(2,024)", -2024.0), ("2,024", 2024.0),
             ("(404,256)", -404256.0)]
    passed = 0
    for token, want in cases:
        passed += check(f"to_number({token!r})", to_number(token), want)
    passed += check("bracketed year is not a money cell",
                    is_money_cell("(2024"), False)
    return passed, len(cases) + 1


def test_markdown_right_anchored_columns():
    """Real Unlimited-OCR output from a Companies House balance sheet.

    The model does not emit the same number of cells on every row. Here the
    fixed assets line has an empty spacer before the figures and the debtors
    line has one after them, so counting columns from the left put the same
    year's figures in different places and the year heading matched only
    one of them. Debtors still came out right, but only because the picker
    fell back to the leftmost figure when nothing matched, which is luck
    rather than logic and would break on a filing that leads with the
    comparative.

    Counting from the right fixes it, because a financial table is right
    aligned by construction. Both rows below must land on the same column
    as the 2025 heading."""
    page = (
        "<table>"
        "<tr><td></td><td>Notes</td><td>\u00a3</td><td>2025\u00a3</td>"
        "<td>2024\u00a3</td></tr>"
        "<tr><td>Tangible Assets</td><td>4</td><td></td><td>684,701</td>"
        "<td>687,352</td></tr>"
        "<tr><td>Debtors</td><td>6</td><td>493,712</td><td>517,202</td>"
        "<td></td></tr>"
        "<tr><td>SHAREHOLDERS&#x27; FUNDS</td><td></td><td></td>"
        "<td>658,163</td><td>645,205</td></tr>"
        "</table>")
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 3, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]

    years = fields.year_columns(rows)
    current = fields.current_year_column(years)
    passed = 0
    passed += check("ragged table, 2025 column found",
                    current[0] if current else None, 2025)

    # The two figure rows have different cell counts. Both current year
    # figures must still sit on the same x.
    xs = {}
    for row in rows:
        label = row.text.split()[0] if row.text.split() else ""
        if label in ("Tangible", "Debtors"):
            xs[label] = sorted(c.x for c in row.cells)[0]
    passed += check("ragged rows share a current year column",
                    len(set(xs.values())) == 1 and len(xs) == 2, True)

    debtors = fields.find_debtors(pages)
    passed += check("ragged table debtors by column not fallback",
                    debtors["value"] if debtors else None, 493712.0)
    passed += check("ragged table debtors year",
                    debtors["year"] if debtors else None, 2025)
    passed += check("html entity decoded",
                    "SHAREHOLDERS' FUNDS" in pages[0]["text"], True)
    return passed, 5


def test_markdown_html_table():
    """Some pages come back as HTML rather than pipe tables. Same treatment,
    because the column index is all the row model needs."""
    page = ("<table><tr><td></td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Turnover</td><td>1,240,900</td><td>1,118,004</td></tr>"
            "</table>")
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 0, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]
    turnover = fields.find_turnover(pages)
    passed = 0
    passed += check("html table turnover",
                    turnover["value"] if turnover else None, 1240900.0)
    passed += check("html table picks 2025 not 2024",
                    turnover["year"] if turnover else None, 2025)
    return passed, 2


def test_money_cell():
    """...but a note reference is not a money cell, which is what stops
    `Debtors 6 493,712` returning 6."""
    cases = [("6", False), ("10", False), ("2025", False), ("2024", False),
             ("493,712", True), ("312.365", True), ("1,022,621", True),
             ("(404,256", True), ("434135", True)]
    passed = 0
    for token, want in cases:
        passed += check(f"is_money_cell({token!r})", is_money_cell(token), want)
    return passed, len(cases)


def test_no_profit_and_loss():
    """The bracketed subsection in 'section 444(1)' broke the old pattern."""
    yes = [
        "The company has taken advantage of section 444(1) of the Companies "
        "Act 2006 and opted not to deliver to the registrar a copy of the "
        "company's Profit and Loss Account.",
        "prepared in accordance with the provisions applicable to companies "
        "subject to the small companies regime",
        "The profit and loss account has not been delivered.",
        "These are filleted accounts.",
    ]
    no = [
        "The consolidated income statement is set out on page 123.",
        "Turnover is measured at the fair value of the consideration.",
    ]
    passed = 0
    for text in yes:
        passed += check(f"no_pl {text[:38]!r}...",
                        fields.no_profit_and_loss(text), True)
    for text in no:
        passed += check(f"no_pl {text[:38]!r}...",
                        fields.no_profit_and_loss(text), False)
    return passed, len(yes) + len(no)


def test_headcount_sentence():
    """The comparative in brackets carries a year with it."""
    cases = [
        ("Average number of employees, including directors, during the year "
         "was: 32 (2024: 30)", 32),
        ("The average number of persons employed during the year was 2,006 "
         "(2024: 1,988)", 2006),
        ("Average monthly number of employees during the period was 7", 7),
        ("Average number of emp1oyees during the year was: 14 (2024: 12)", 14),
    ]
    passed = 0
    for sentence, want in cases:
        got = fields._first_headcount(sentence)
        passed += check(f"headcount {sentence[:38]!r}...", got, want)
    return passed, len(cases)


def test_turnover_estimate():
    """Trade debtors is the right basis. Estimating off the balance sheet
    Debtors total on the sample filing overstates by 58%, because 115,847
    of it is a loan to a former director."""
    cases = [
        # trade debtors present, total also present: use trade
        ((312365.0, 493712.0), 1561825.0, "trade debtors"),
        # trade debtors missing: fall back to the total and flag it
        ((None, 493712.0), 2468560.0, "total debtors"),
        # nothing to work from
        ((None, None), None, None),
        # too small to mean anything
        ((None, 400.0), None, None),
    ]
    passed = 0
    for (trade, total), want_value, want_basis in cases:
        got = estimate.estimate_turnover(trade, total, multiple=5.0)
        passed += check(f"estimate(trade={trade}, total={total})",
                        got["value"] if got else None, want_value)
        passed += check("  basis", got["basis"] if got else None, want_basis)
    return passed, len(cases) * 2


def test_estimate_never_overwrites():
    """A filed turnover must survive untouched, and an estimate must never
    land in the turnover field."""
    import pipeline
    filed = pipeline.add_estimate({
        "turnover": 372104000.0, "debtors": 32273000.0,
        "trade_debtors": None})
    missing = pipeline.add_estimate({
        "turnover": None, "debtors": 493712.0, "trade_debtors": 312365.0})
    passed = 0
    passed += check("filed turnover untouched", filed["turnover"], 372104000.0)
    passed += check("filed gets no estimate", filed["turnover_estimate"], None)
    passed += check("filed source", filed["turnover_source"], "filed")
    passed += check("observed multiple recorded",
                    round(filed["observed_debtor_multiple"], 1), 11.5)
    passed += check("multiple is hardcoded at 5", estimate.MULTIPLE, 5.0)
    # ratios from misread filings must not reach the calibration summary
    passed += check("implausible ratio rejected (t=2024, d=1.671bn)",
                    estimate.observed_multiple(2024.0, 1671000000.0), None)
    passed += check("plausible ratio kept (t=325.8m, d=61.4m)",
                    round(estimate.observed_multiple(
                        325851000.0, 61448000.0), 1), 5.3)
    passed += check("missing turnover stays null", missing["turnover"], None)
    passed += check("estimate populated",
                    missing["turnover_estimate"], 1561825.0)
    passed += check("estimate source",
                    missing["turnover_source"],
                    "estimated from trade debtors")
    return passed, 10


def test_sample_filing():
    """End to end against the real filing, if it is present."""
    if not os.path.exists(SAMPLE_PDF):
        print(f"  skip  sample not found at {SAMPLE_PDF}")
        return 0, 0
    import pipeline
    raw = open(SAMPLE_PDF, "rb").read()
    result = pipeline.read_filing_pdf(raw)
    flat = pipeline._flatten(result["found"])
    passed = 0
    for key, want in EXPECTED.items():
        passed += check(f"sample filing {key}", flat[key], want)
    passed += check("sample filing no_profit_and_loss",
                    result["no_profit_and_loss"], True)
    passed += check("sample filing trade_debtors", flat["trade_debtors"],
                    312365.0)
    passed += check("sample filing net_assets", flat["net_assets"], 658163.0)
    passed += check("sample filing total_assets", flat["total_assets"],
                    1662593.0)
    passed += check("sample filing total_liabilities",
                    flat["total_liabilities"], 1004430.0)
    record = pipeline.add_estimate(flat)
    passed += check("sample filing turnover_estimate",
                    record["turnover_estimate"], 1561825.0)
    return passed, len(EXPECTED) + 6


def test_balance_sheet_markdown():
    """Net assets, total assets and total liabilities from a balance sheet.

    Net current assets must never be read as net assets, and total assets is
    worked out from the two lines every statutory balance sheet prints."""
    page = (
        "# Balance sheet\n\n"
        "| | Note | 2025 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Net current assets | | 573,636 | 552,927 |\n"
        "| Total assets less current liabilities | | 1,258,337 | 1,240,279 |\n"
        "| Creditors: amounts falling due within one year | 7 | (404,256) | (469,694) |\n"
        "| Net assets | | 658,163 | 645,205 |\n"
    )
    rows = markdown_rows.rows_from_markdown(page)
    pages = [{"index": 3, "rows": rows,
              "text": markdown_rows.markdown_text(page)}]
    net = fields.find_net_assets(pages)
    assets = fields.find_total_assets(pages)
    liab = fields.find_total_liabilities(pages, assets, net)
    passed = 0
    passed += check("net assets, not net current assets", net["value"], 658163.0)
    passed += check("total assets = TALCL + creditors < 1 year",
                    assets["value"], 1662593.0)
    passed += check("total liabilities = assets - net assets",
                    liab["value"], 1004430.0)

    deficit = (
        "# Balance sheet\n\n"
        "| | 2025 | 2024 |\n| --- | --- | --- |\n"
        "| Net liabilities | 12,000 | 8,000 |\n"
    )
    rows = markdown_rows.rows_from_markdown(deficit)
    pages = [{"index": 2, "rows": rows,
              "text": markdown_rows.markdown_text(deficit)}]
    passed += check("net liabilities is negative",
                    fields.find_net_assets(pages)["value"], -12000.0)
    return passed, 4


def test_balance_sheet_ixbrl():
    """FRS 102 iXBRL: no Assets tag, so total assets is fixed + current, and
    a dimensioned Equity fact (one class of share capital) is ignored."""
    import ixbrl
    doc = (b'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
           b'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
           b'xmlns:xbrldi="http://xbrl.org/2006/xbrldi"><body>'
           b'<xbrli:context id="c1"><xbrli:period><xbrli:instant>2025-01-31'
           b'</xbrli:instant></xbrli:period></xbrli:context>'
           b'<xbrli:context id="c0"><xbrli:period><xbrli:instant>2024-01-31'
           b'</xbrli:instant></xbrli:period></xbrli:context>'
           b'<xbrli:context id="d1"><xbrli:entity><xbrli:segment>'
           b'<xbrldi:explicitMember dimension="x">core:ShareCapital'
           b'</xbrldi:explicitMember></xbrli:segment></xbrli:entity>'
           b'<xbrli:period><xbrli:instant>2025-01-31</xbrli:instant>'
           b'</xbrli:period></xbrli:context>'
           b'<ix:nonFraction name="core:FixedAssets" contextRef="c1">684,701'
           b'</ix:nonFraction>'
           b'<ix:nonFraction name="core:CurrentAssets" contextRef="c1">977,892'
           b'</ix:nonFraction>'
           b'<ix:nonFraction name="core:CurrentAssets" contextRef="c0">999,999'
           b'</ix:nonFraction>'
           b'<ix:nonFraction name="core:NetAssetsLiabilities" contextRef="c1">'
           b'658,163</ix:nonFraction>'
           b'<ix:nonFraction name="core:Equity" contextRef="d1">100'
           b'</ix:nonFraction>'
           b'</body></html>')
    r = ixbrl.parse(doc)
    passed = 0
    passed += check("ixbrl net assets", r["net_assets"], 658163.0)
    passed += check("ixbrl total assets, current period only",
                    r["total_assets"], 1662593.0)
    passed += check("ixbrl total liabilities", r["total_liabilities"],
                    1004430.0)
    return passed, 3


def test_holding_company_ixbrl():
    """Afan Group Ltd (12616390), 31 Jan 2025, as filed. Its only fixed asset
    is tagged FixedAssetInvestments with no FixedAssets total, which made
    total assets come out as current assets alone (68,769). Its debtors are
    all owed by subsidiaries, so no turnover estimate may be made from them."""
    import ixbrl
    import pipeline
    ctx = (b'<xbrli:context id="c1"><xbrli:period><xbrli:instant>2025-01-31'
           b'</xbrli:instant></xbrli:period></xbrli:context>'
           b'<xbrli:context id="w1"><xbrli:entity><xbrli:segment>'
           b'<xbrldi:explicitMember dimension="x">core:WithinOneYear'
           b'</xbrldi:explicitMember></xbrli:segment></xbrli:entity>'
           b'<xbrli:period><xbrli:instant>2025-01-31</xbrli:instant>'
           b'</xbrli:period></xbrli:context>')
    def nf(name, c, v):
        return (f'<ix:nonFraction name="core:{name}" contextRef="{c}">{v}'
                f'</ix:nonFraction>').encode()
    body = b"".join([
        nf("FixedAssetInvestments", "c1", "578,000"),
        nf("Debtors", "c1", "56,169"), nf("CurrentAssets", "c1", "68,769"),
        nf("Creditors", "w1", "30,328"),
        nf("NetCurrentAssetsLiabilities", "c1", "38,441"),
        nf("TotalAssetsLessCurrentLiabilities", "c1", "616,441"),
        nf("Equity", "c1", "616,441"),
        nf("AmountsOwedByGroupUndertakingsParticipatingInterests", "c1",
           "56,169")])
    doc = (b'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
           b'xmlns:xbrli="http://www.xbrl.org/2003/instance" '
           b'xmlns:xbrldi="http://xbrl.org/2006/xbrldi"><body>' + ctx + body +
           b'</body></html>')
    r = ixbrl.parse(doc)
    rec = pipeline.add_estimate({"turnover": None, "debtors": r["debtors"],
                                 "trade_debtors": None,
                                 "debtors_intercompany":
                                     r["debtors_intercompany"]})
    passed = 0
    passed += check("holding co net assets", r["net_assets"], 616441.0)
    passed += check("holding co total assets includes investments",
                    r["total_assets"], 646769.0)
    passed += check("holding co total liabilities", r["total_liabilities"],
                    30328.0)
    passed += check("no estimate from intercompany debtors",
                    rec["turnover_estimate"], None)
    return passed, 4


def main():
    total_passed = total_run = 0
    for name, func in [
        ("number repair", test_separator_repair),
        ("light repair for the vision model", test_light_repair_mode),
        ("markdown to rows", test_markdown_rows),
        ("markdown without a year header", test_markdown_without_year_header),
        ("markdown html table", test_markdown_html_table),
        ("ragged model tables", test_markdown_right_anchored_columns),
        ("bracketed years", test_bracketed_year_is_not_a_figure),
        ("scale applied consistently", test_scale_applied_consistently),
        ("tabulated headcount", test_employees_tabulated),
        ("money cell detection", test_money_cell),
        ("profit and loss withheld", test_no_profit_and_loss),
        ("headcount sentence", test_headcount_sentence),
        ("turnover estimate", test_turnover_estimate),
        ("estimate stays separate", test_estimate_never_overwrites),
        ("balance sheet from markdown", test_balance_sheet_markdown),
        ("balance sheet from ixbrl", test_balance_sheet_ixbrl),
        ("holding company ixbrl", test_holding_company_ixbrl),
        ("sample filing end to end", test_sample_filing),
    ]:
        print(f"\n{name}")
        passed, run = func()
        total_passed += passed
        total_run += run

    print(f"\n{'=' * 60}\n{total_passed}/{total_run} passed")
    return 0 if total_passed == total_run else 1


if __name__ == "__main__":
    sys.exit(main())