"""
Companies House accounts extractor.

    pip install -r requirements.txt
    $env:CH_API_KEY = "your key"
    python app.py

Endpoints
---------
POST /v1/company   company number in, turnover, debtors, headcount and the
                   balance sheet (net assets, total assets, total liabilities)
                   out. Results are cached per company for CACHE_DAYS.
POST /v1/ocr       read any PDF, returns page markdown
POST /v1/ixbrl     parse iXBRL you already have
GET  /health

Every figure comes back with the line it was read from, so a wrong number
can be traced to a page rather than argued about.
"""

import base64
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import ch_client
import ixbrl
import pipeline
import reader

API_TOKEN = os.getenv("API_TOKEN", "")
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "127.0.0.1")

ch = ch_client.CompaniesHouse(os.getenv("CH_API_KEY", ""))
app = FastAPI(title="companies house accounts extractor")

# OCR is CPU bound. On a small VPS two filings read at once take as long as
# reading them one after the other, and both risk the n8n timeout, so jobs
# queue here instead. iXBRL filings are quick and queue for seconds at most.
MAX_JOBS = int(os.getenv("MAX_JOBS", "1"))
_jobs = threading.BoundedSemaphore(MAX_JOBS)

# A parent company's group accounts get asked for once per subsidiary that is
# researched. Accounts are filed once a year, so a cache saves real OCR time.
CACHE_DAYS = float(os.getenv("CACHE_DAYS", "30"))
CACHE_PATH = os.path.join(os.getenv("CACHE_DIR", "."), "financials_cache.sqlite")
_cache_lock = threading.Lock()
CACHEABLE = {"ok", "no_figures_found", "no_accounts_filed", "company_not_found"}


def _db():
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS cache (number TEXT PRIMARY KEY, "
                "saved REAL, body TEXT)")
    return con


def cache_get(number: str) -> Optional[Dict[str, Any]]:
    if CACHE_DAYS <= 0:
        return None
    with _cache_lock:
        con = _db()
        try:
            row = con.execute("SELECT saved, body FROM cache WHERE number=?",
                              (number,)).fetchone()
        finally:
            con.close()
    if not row or time.time() - row[0] > CACHE_DAYS * 86400:
        return None
    body = json.loads(row[1])
    body["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row[0]))
    return body


def cache_put(number: str, body: Dict[str, Any]) -> None:
    if CACHE_DAYS <= 0 or body.get("status") not in CACHEABLE:
        return
    with _cache_lock:
        con = _db()
        try:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (number, time.time(), json.dumps(body, default=str)))
            con.commit()
        finally:
            con.close()


def summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The handful of fields the research sheet needs, in one flat object.

    turnover is only ever a filed figure. An estimate stays in its own
    field so nobody quotes it to a seller as fact."""
    return {
        "company_number": rec.get("company_number"),
        "status": rec.get("status"),
        "route": rec.get("route"),
        "accounts_type": rec.get("accounts_type"),
        "consolidated": rec.get("accounts_type") == "group",
        "made_up_date": rec.get("made_up_date"),
        "filing_date": rec.get("filing_date"),
        "turnover": rec.get("turnover"),
        "turnover_source": rec.get("turnover_source"),
        "turnover_on_record": rec.get("turnover_on_record"),
        "turnover_estimate": rec.get("turnover_estimate"),
        "turnover_estimate_basis": rec.get("turnover_estimate_basis"),
        "turnover_estimate_confidence": rec.get("turnover_estimate_confidence"),
        "observed_debtor_multiple": rec.get("observed_debtor_multiple"),
        "debtors": rec.get("debtors"),
        "trade_debtors": rec.get("trade_debtors"),
        "employee_size": rec.get("employee_size"),
        "net_assets": rec.get("net_assets"),
        "total_assets": rec.get("total_assets"),
        "total_liabilities": rec.get("total_liabilities"),
        "total_assets_working": rec.get("total_assets_working"),
        "total_liabilities_working": rec.get("total_liabilities_working"),
        "balance_check": rec.get("balance_check"),
        "balance_check_detail": rec.get("balance_check_detail"),
        "pages_ocred": rec.get("pages_ocred") or 0,
    }


@app.middleware("http")
async def guard(request, call_next):
    """Anything on a public URL needs a token. Without one anybody who
    guesses the address can burn your Companies House rate limit."""
    if API_TOKEN and request.url.path.startswith("/v1/"):
        if request.headers.get("x-auth-token") != API_TOKEN:
            return JSONResponse({"error": "bad or missing x-auth-token"},
                                status_code=401)
    return await call_next(request)


class CompanyRequest(BaseModel):
    company_number: str
    company_name: Optional[str] = None
    max_filings: int = 3
    allow_ocr: bool = True
    return_text: bool = False
    refresh: bool = False          # ignore the cache and read the filing again


class DocumentRef(BaseModel):
    type: Optional[str] = None
    document_url: Optional[str] = None
    document_base64: Optional[str] = None


class OcrRequest(BaseModel):
    model: Optional[str] = None
    document: DocumentRef
    pages: Optional[Any] = None


class IxbrlRequest(BaseModel):
    content: Optional[str] = None
    content_base64: Optional[str] = None


_inflight: Dict[str, threading.Lock] = {}
_inflight_guard = threading.Lock()


def _company_lock(number: str) -> threading.Lock:
    with _inflight_guard:
        return _inflight.setdefault(number, threading.Lock())


@app.post("/v1/company")
def company(req: CompanyRequest) -> Dict[str, Any]:
    number = ch_client.normalise(req.company_number)
    if not req.refresh and not req.return_text:
        hit = cache_get(number)
        if hit is not None:
            return hit
    # If a proxy (Cloudflare tunnel, ~100 s) cut the caller off and n8n retries,
    # the retry waits here for the first read to finish and gets its cached
    # result, instead of starting the same OCR job a second time.
    with _company_lock(number):
        if not req.refresh and not req.return_text:
            hit = cache_get(number)
            if hit is not None:
                return hit
        started = time.time()
        with _jobs:
            try:
                rec = pipeline.company(ch, number, req.company_name,
                                       req.max_filings, req.allow_ocr,
                                       req.return_text)
            except RuntimeError as exc:
                raise HTTPException(502, str(exc))
        rec["secs"] = round(time.time() - started, 1)
        rec["summary"] = summary(rec)
        print(f"{number}: {rec.get('status')} via {rec.get('route')} "
              f"in {rec['secs']}s", flush=True)
        if not req.return_text:
            cache_put(number, rec)
        return rec


def wanted_pages(spec, total: int) -> Optional[List[int]]:
    if spec is None:
        return None
    if isinstance(spec, int):
        return [spec] if 0 <= spec < total else []
    if isinstance(spec, list):
        return [p for p in spec if 0 <= p < total]
    text = str(spec)
    if "-" in text:
        first, last = text.split("-", 1)
        return [p for p in range(int(first), int(last) + 1) if 0 <= p < total]
    value = int(text)
    return [value] if 0 <= value < total else []


@app.post("/v1/ocr")
def ocr(req: OcrRequest) -> Dict[str, Any]:
    if req.document.document_base64:
        try:
            raw = base64.b64decode(req.document.document_base64, validate=True)
        except Exception as exc:
            raise HTTPException(400, f"bad base64: {exc}")
    elif req.document.document_url:
        try:
            with httpx.Client(follow_redirects=True, timeout=180.0) as client:
                response = client.get(req.document.document_url)
                response.raise_for_status()
                raw = response.content
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"could not fetch document: {exc}")
    else:
        raise HTTPException(400, "supply document_url or document_base64")

    import pymupdf
    try:
        probe = pymupdf.open(stream=raw, filetype="pdf")
        total = probe.page_count
        probe.close()
    except Exception as exc:
        raise HTTPException(422, f"could not open pdf: {exc}")

    try:
        result = reader.read_pdf(raw, allow_ocr=True,
                                 page_list=wanted_pages(req.pages, total))
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # markdown carries the model's own output when the vision engine is in
    # use, and the flattened row text otherwise, so the n8n nodes downstream
    # see the same shape whichever engine read the page. The detection
    # markers are stripped here: they are useful for debugging a page but
    # they are noise to anything reading the markdown, including an LLM.
    import markdown_rows
    return {
        "pages": [{"index": p["index"],
                   "markdown": markdown_rows.strip_markers(
                       p.get("markdown") or p["text"]),
                   "text": p["text"],
                   "source": p["source"], "images": []}
                  for p in result["pages"]],
        "model": f"pymupdf-{result.get('engine', 'tesseract')}-layout",
        "usage_info": {
            "pages_processed": len(result["pages"]),
            "pages_ocred": result["pages_ocred"],
            "engine": result.get("engine"),
            "doc_size_bytes": len(raw),
            "ocr_errors": result["ocr_errors"],
        },
    }


@app.post("/v1/ixbrl")
def parse_ixbrl(req: IxbrlRequest) -> Dict[str, Any]:
    if req.content:
        raw = req.content.encode("utf-8", "ignore")
    elif req.content_base64:
        try:
            raw = base64.b64decode(req.content_base64, validate=True)
        except Exception as exc:
            raise HTTPException(400, f"bad base64: {exc}")
    else:
        raise HTTPException(400, "supply content or content_base64")

    if raw[:2] == b"PK":
        inner = ixbrl.ixbrl_from_zip(raw)
        if inner is None:
            raise HTTPException(422, "zip contained no iXBRL")
        raw = inner
    try:
        return ixbrl.parse(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/health")
def health() -> Dict[str, Any]:
    """Says which engine is configured and whether it actually answers.

    engine_ready False with an engine_error string is the useful case: the
    service is up but the GPU box behind it is not, and you want to know
    that from a health check rather than from a failed batch.
    """
    engine = reader.OCR_ENGINE
    ready, detail = True, None
    try:
        detail = reader.check_ocr()
    except RuntimeError as exc:
        ready, detail = False, str(exc).splitlines()[0]
    return {"ok": True,
            "ch_key_set": bool(os.getenv("CH_API_KEY")),
            "auth_required": bool(API_TOKEN),
            "engine": engine,
            "engine_ready": ready,
            "engine_detail": detail if ready else None,
            "engine_error": None if ready else detail,
            "scout_engine": os.getenv("SCOUT_ENGINE", engine),
            "tesseract": detail if ready and engine == "tesseract" else None,
            "read_dpi": reader.READ_DPI,
            "max_jobs": MAX_JOBS,
            "cache_days": CACHE_DAYS,
            "psm": reader.OCR_PSM}


if __name__ == "__main__":
    print(f"listening on http://{HOST}:{PORT}")
    print("auth token " + ("set" if API_TOKEN
                           else "NOT SET, /v1/ endpoints are open"))
    print(f"ocr engine {reader.OCR_ENGINE} at {reader.READ_DPI} dpi")
    try:
        print(f"  {reader.check_ocr()}")
    except RuntimeError as exc:
        print("\n!!! " + str(exc) + "\n")
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=75)
