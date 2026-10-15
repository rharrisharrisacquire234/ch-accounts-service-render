"""
One interface, two OCR engines, so nothing downstream has to know or care.

An engine takes page images and returns rows. That is the whole contract.
Tesseract gets there through word boxes, Unlimited-OCR through parsed
markdown, and both come out the other side as the same Row objects that
fields.py has always worked on.

    OCR_ENGINE=tesseract     the original path, CPU, no GPU needed
    OCR_ENGINE=unlimited     baidu/Unlimited-OCR, needs a GPU somewhere
    SCOUT_ENGINE=tesseract   optional, cheap first look with the other one

That last one is worth understanding before you run a batch. On a 195 page
group filing the reader takes a cheap look at pages to find where the
statements are, then reads those properly. With Tesseract a scout page costs
about a second. With a vision model a scout page costs almost as much as a
real read, so scouting with the model can double the bill on large filings.
Setting SCOUT_ENGINE=tesseract keeps the cheap search and spends the model
only on the pages that matter. Small filings are read whole and never scout,
so this changes nothing for a ten page set of small company accounts.
"""

import concurrent.futures as cf
import os
from typing import List, Optional, Tuple

from PIL import Image

from layout import Row, page_rows, page_text

OCR_PSM = int(os.getenv("OCR_PSM", "6"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
UNLIMITED_WORKERS = int(os.getenv("UNLIMITED_WORKERS", "4"))

# (page index, rows, raw model output or None, error or None)
Result = Tuple[int, List[Row], Optional[str], Optional[str]]


class TesseractEngine:
    """The original path. Word boxes in, rows out."""

    name = "tesseract"
    needs_gpu = False

    def __init__(self):
        self._version = None

    def check(self) -> str:
        if self._version:
            return self._version
        import shutil
        import pytesseract
        cmd = os.getenv("TESSERACT_CMD", "")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        elif os.name == "nt" and not shutil.which("tesseract"):
            for guess in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                          r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                if os.path.exists(guess):
                    pytesseract.pytesseract.tesseract_cmd = guess
                    break
        try:
            self._version = str(pytesseract.get_tesseract_version())
        except Exception as exc:
            raise RuntimeError(
                f"Tesseract is not callable: {exc}\n"
                "Companies House serves most accounts as image only scans, "
                "so OCR is required.\n"
                "  winget install UB-Mannheim.TesseractOCR\n"
                '  $env:TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\'
                'tesseract.exe"')
        return self._version

    @property
    def workers(self) -> int:
        return OCR_WORKERS

    def run(self, images: List[Tuple[int, Image.Image]]) -> List[Result]:
        if not images:
            return []
        with cf.ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
            return list(pool.map(self._one, images))

    @staticmethod
    def _one(item) -> Result:
        index, img = item
        try:
            rows = page_rows(img, OCR_PSM, OCR_LANG)
            return index, rows, page_text(rows), None
        except Exception as exc:
            return index, [], None, str(exc)


class UnlimitedEngine:
    """baidu/Unlimited-OCR. Markdown in, the same rows out.

    The model reads a page as a document rather than as a bag of words, so
    a financial table comes back as a pipe table with the columns intact.
    markdown_rows.py turns that back into the row model, which means the
    year column logic, the separator repairs and every test carry over
    untouched.
    """

    name = "unlimited"
    needs_gpu = True

    def __init__(self):
        import unlimited_client
        self._client = unlimited_client.build()
        self._checked = None

    def check(self) -> str:
        if self._checked is None:
            self._checked = self._client.check()
        return self._checked

    @property
    def workers(self) -> int:
        # Local mode serialises anyway, so asking for concurrency there just
        # queues requests behind a lock and makes the timings confusing.
        from unlimited_client import UnlimitedLocal
        if isinstance(self._client, UnlimitedLocal):
            return 1
        return UNLIMITED_WORKERS

    def run(self, images: List[Tuple[int, Image.Image]]) -> List[Result]:
        if not images:
            return []
        workers = max(1, min(self.workers, len(images)))
        if workers == 1:
            return [self._one(item) for item in images]
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._one, images))

    def _one(self, item) -> Result:
        from markdown_rows import rows_from_markdown
        index, img = item
        try:
            raw = self._client.read(img)
            return index, rows_from_markdown(raw), raw, None
        except Exception as exc:
            return index, [], None, str(exc)


_ENGINES = {}


def get_engine(name: Optional[str] = None):
    """Engines are cached because both of them hold something expensive:
    a resolved tesseract path, or an http pool and possibly model weights."""
    name = (name or os.getenv("OCR_ENGINE", "tesseract")).lower().strip()
    if name in ("unlimited-ocr", "unlimited_ocr", "baidu", "vlm"):
        name = "unlimited"
    if name not in ("tesseract", "unlimited"):
        raise RuntimeError(f"unknown OCR_ENGINE {name!r}, "
                           "use tesseract or unlimited")
    if name not in _ENGINES:
        _ENGINES[name] = (UnlimitedEngine() if name == "unlimited"
                          else TesseractEngine())
    return _ENGINES[name]
