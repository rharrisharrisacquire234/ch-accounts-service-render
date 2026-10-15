"""
Reading a filing end to end, spending OCR only where it will pay.

Page selection is the thing that decides whether this works on a large
filing. A set of group accounts can run to 200 pages, and the consolidated
income statement sits somewhere in the middle, well past any "first N pages"
window. Scanning the front of the document finds the strategic report and
nothing else.

So instead of a fixed window we scout: read a handful of pages cheaply,
look for the words that mark the statements we want, then read those pages
properly. On a ten page small company filing the scout covers the whole
document anyway and costs nothing extra.

Which OCR engine does the reading is now a setting rather than a rewrite.
Everything in here works in pages and rows, and ocr_engine.py is the only
file that knows whether those rows came from Tesseract word boxes or from
Unlimited-OCR markdown.
"""

import os
import re
from typing import Any, Dict, List, Optional

import pymupdf

import figures
import ocr_engine
from layout import page_image, page_text, rows_from_text_layer

OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").lower().strip()
_IS_VLM = OCR_ENGINE not in ("tesseract", "")

MIN_CHARS_PER_PAGE = int(os.getenv("MIN_CHARS_PER_PAGE", "120"))
# A vision model wants a little more resolution than Tesseract does, and it
# is not paying per pixel in the same way, so the defaults differ by engine.
READ_DPI = int(os.getenv("READ_DPI", "300" if _IS_VLM else "200"))
SCOUT_DPI = int(os.getenv("SCOUT_DPI", "150"))
OCR_PSM = int(os.getenv("OCR_PSM", "6"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "45"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "400"))
SMALL_DOC_PAGES = int(os.getenv("SMALL_DOC_PAGES", "24"))

# Pages worth reading properly. Deliberately broad: a page that is wrongly
# included costs a second, a page that is wrongly excluded costs the figure.
WANTED = re.compile(
    r"t[uv][rm]n?[o0]ver|tum[o0]ver|reven[uv]e|"
    r"debt[o0]rs?|receivable|"
    r"emp[l1i][o0]yee|personnel|headcount|staff\s+cost|"
    r"profit\s+and\s+loss|income\s+statement|balance\s+sheet|"
    r"financial\s+position|average\s+number|"
    r"section\s+444|filleted|abridged",
    re.I,
)

# Contents pages list the statements against their page numbers. Reading it
# is one page of OCR that can save a hundred.
CONTENTS_LINE = re.compile(
    r"(profit\s+and\s+loss|income\s+statement|balance\s+sheet|"
    r"financial\s+position|notes\s+to\s+the)"
    r"[^\d\n]{0,60}(\d{1,3})(?:\s*[-\u2013\u2014]\s*(\d{1,3}))?",
    re.I,
)


def _apply_repair_mode():
    """Tesseract needs the letter to digit repairs. A vision model does not,
    and running them on clean output is a small risk for no gain. Set
    FIGURE_REPAIR explicitly to override the engine's default."""
    figures.set_repair_mode(
        os.getenv("FIGURE_REPAIR", "light" if _IS_VLM else "full"))


_apply_repair_mode()


def set_engine(name: str, dpi: Optional[int] = None) -> None:
    """Switch engine at runtime. Used by cli.py --engine so the same PDF can
    be read both ways in one sitting and the figures compared."""
    global OCR_ENGINE, _IS_VLM, READ_DPI
    OCR_ENGINE = name.lower().strip()
    _IS_VLM = OCR_ENGINE not in ("tesseract", "")
    if dpi:
        READ_DPI = dpi
    elif "READ_DPI" not in os.environ:
        READ_DPI = 300 if _IS_VLM else 200
    _apply_repair_mode()


def read_engine():
    return ocr_engine.get_engine(OCR_ENGINE)


def scout_engine():
    """Scouting with a vision model costs nearly as much as reading with it,
    so SCOUT_ENGINE lets you keep the cheap search on Tesseract. If that is
    asked for and Tesseract is not installed, fall back rather than fail."""
    name = os.getenv("SCOUT_ENGINE", OCR_ENGINE)
    try:
        engine = ocr_engine.get_engine(name)
        engine.check()
        return engine
    except Exception:
        return read_engine()


def check_ocr() -> str:
    """Fail once, loudly, at the top rather than silently on every page."""
    return read_engine().check()


# The old name, kept so nothing that imported it has to change.
check_tesseract = check_ocr


def engine_name() -> str:
    return read_engine().name


def scout_order(page_count: int) -> List[int]:
    """Which pages to look at first when hunting for the statements.

    Small filings get read whole. On anything larger, the statements sit
    after the narrative reports, so start around the middle and work out,
    but always look at the first two pages because that is where a contents
    page lives.
    """
    if page_count <= SMALL_DOC_PAGES:
        return list(range(page_count))
    middle = int(page_count * 0.45)
    order = [0, 1]
    order += list(range(middle, page_count))
    order += list(range(2, middle))
    seen, out = set(), []
    for p in order:
        if p not in seen and 0 <= p < page_count:
            seen.add(p)
            out.append(p)
    return out


def contents_targets(text: str, page_count: int) -> List[int]:
    """Turn a contents page into page indices to read.

    Printed page numbers usually differ from PDF page indices by the number
    of front matter pages, so this returns a window either side rather than
    an exact page.
    """
    targets = []
    for match in CONTENTS_LINE.finditer(text):
        for group in (match.group(2), match.group(3)):
            if not group:
                continue
            printed = int(group)
            if 0 < printed <= page_count:
                for offset in (-1, 0, 1, 2):
                    page = printed + offset
                    if 0 <= page < page_count:
                        targets.append(page)
    return sorted(set(targets))


def read_pdf(raw: bytes, allow_ocr: bool = True,
             page_list: Optional[List[int]] = None,
             max_ocr: Optional[int] = None,
             stop_when=None) -> Dict[str, Any]:
    """Return the row model for the pages that matter, plus what it cost.

    stop_when is called with the pages read so far after each batch. Return
    True from it and reading stops there. That is what keeps the cost down:
    on a filing where the figures turn up on page 3 there is no reason to
    read pages 4 to 200.
    """
    if not raw.startswith(b"%PDF"):
        head = raw[:120].decode("utf-8", "ignore")
        raise ValueError(f"not a PDF ({len(raw)} bytes): {head}")

    pdf = pymupdf.open(stream=raw, filetype="pdf")
    try:
        return _read(pdf, allow_ocr, page_list, max_ocr, stop_when)
    finally:
        pdf.close()          # the previous version never closed these


def _chunks(items, size):
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _read(pdf, allow_ocr, page_list, max_ocr, stop_when=None) -> Dict[str, Any]:
    page_count = pdf.page_count
    cap = max_ocr if max_ocr is not None else MAX_OCR_PAGES
    pages: Dict[int, Dict[str, Any]] = {}
    ocr_used = 0
    errors: List[str] = []

    def add(index: int, rows, source: str, markdown: Optional[str] = None):
        text = page_text(rows)
        pages[index] = {"index": index, "rows": rows, "text": text,
                        "markdown": markdown or text, "source": source}

    # ---- pass one: free text layer, wherever there is one ----
    candidates = page_list if page_list is not None else \
        list(range(min(page_count, MAX_PDF_PAGES)))
    needs_ocr = []
    for index in candidates:
        if not 0 <= index < page_count:
            continue
        page = pdf.load_page(index)
        if len(page.get_text("text").strip()) >= MIN_CHARS_PER_PAGE:
            add(index, rows_from_text_layer(page), "text_layer")
        else:
            needs_ocr.append(index)

    if not allow_ocr or not needs_ocr:
        return _result(pages, page_count, ocr_used, errors, "text_layer")

    engine = read_engine()
    engine.check()

    # ---- pass two: cheap scout to find the pages worth reading ----
    # On a short filing the scout would cover the whole document anyway, so
    # it just doubles the OCR bill. Read those straight through instead.
    if page_list is not None or page_count <= SMALL_DOC_PAGES:
        targets = needs_ocr
    else:
        scout = scout_engine()
        targets, scouted = [], 0
        contents_seen = False
        order = [p for p in scout_order(page_count) if p in needs_ocr]
        for batch in _chunks(order, scout.workers):
            if scouted >= cap:
                break
            images = [(p, page_image(pdf.load_page(p), SCOUT_DPI))
                      for p in batch]
            for index, rows, markdown, err in sorted(scout.run(images),
                                                     key=lambda r: r[0]):
                if err:
                    errors.append(f"p{index}: {err}")
                    continue
                scouted += 1
                ocr_used += 1
                add(index, rows, f"{scout.name}_scout_{SCOUT_DPI}", markdown)
                text = pages[index]["text"]

                if not contents_seen and index <= 2:
                    extra = contents_targets(text, page_count)
                    if extra:
                        contents_seen = True
                        targets.extend(p for p in extra if p in needs_ocr)
                if WANTED.search(text):
                    targets.append(index)
        targets = sorted(set(targets))

    # ---- pass three: reread the pages that matter at full resolution ----
    todo = [p for p in targets if p < page_count][:cap]
    for batch in _chunks(todo, engine.workers):
        images = [(p, page_image(pdf.load_page(p), READ_DPI)) for p in batch]
        for index, rows, markdown, err in sorted(engine.run(images),
                                                 key=lambda r: r[0]):
            if err:
                errors.append(f"p{index}: {err}")
                continue
            ocr_used += 1
            add(index, rows, f"{engine.name}_{READ_DPI}", markdown)
        if stop_when and stop_when([pages[k] for k in sorted(pages)]):
            break

    return _result(pages, page_count, ocr_used, errors, engine.name)


def _result(pages, page_count, ocr_used, errors, engine):
    ordered = [pages[k] for k in sorted(pages)]
    return {
        "pages": ordered,
        "page_count": page_count,
        "pages_read": [p["index"] for p in ordered],
        "pages_ocred": ocr_used,
        "engine": engine,
        "text": "\n".join(p["text"] for p in ordered),
        "ocr_errors": errors[:5],
    }
