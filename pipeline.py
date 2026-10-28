"""
One company in, three figures out, taking the cheapest route that works.

Order of preference, and it matters a great deal:

1. iXBRL, from a zip report package or a bare xhtml. Exact tagged figures,
   about a second, no OCR and no model. Most filings after roughly 2023
   have this.
2. PDF with a real text layer. Free to read, perfectly accurate.
3. PDF that is an image only scan. OCR, targeted at the pages that matter.

Before any of that, the filing description tells you whether turnover can
be on the record at all. A small company filing filleted accounts under
section 444 does not deliver a profit and loss account to the registrar, so
turnover is not there to find. Reporting that honestly is far more useful
than an OCR run that returns nothing after two minutes.
"""

import os
from typing import Any, Dict, List, Optional

import ch_client
import checks
import estimate
import fields
import ixbrl
import reader

MAX_FILINGS = int(os.getenv("MAX_FILINGS", "3"))
EMPTY_BALANCE = {"net_assets": None, "total_assets": None,
                 "total_liabilities": None}


def extract_from_pages(pages: List[Dict[str, Any]],
                       want_turnover: bool) -> Dict[str, Any]:
    turnover = fields.find_turnover(pages) if want_turnover else None
    found = {
        "turnover": turnover,
        "debtors": fields.find_debtors(pages),
        "trade_debtors": fields.find_trade_debtors(pages),
        "employees": fields.find_employees(pages),
        "intercompany_debtors": fields.find_intercompany_debtors(pages),
        **balance_sheet(pages),
    }
    found["balance_check"] = checks.balance_sheet(pages, found)
    return found


def balance_sheet(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    net = fields.find_net_assets(pages)
    assets = fields.find_total_assets(pages, net)
    return {
        "net_assets": net,
        "total_assets": assets,
        "total_liabilities": fields.find_total_liabilities(pages, assets, net),
    }


def add_estimate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fill turnover_estimate when turnover itself is not on the record.

    Deliberately a separate field. An estimate that lands in the turnover
    column is indistinguishable from a filed figure a month later, and
    somebody will end up quoting it to a seller as fact.
    """
    record["turnover_estimate"] = None
    record["turnover_estimate_basis"] = None
    record["turnover_estimate_confidence"] = None
    record["turnover_source"] = "filed" if record.get("turnover") is not None \
        else "not filed"
    record["observed_debtor_multiple"] = estimate.observed_multiple(
        record.get("turnover"), record.get("trade_debtors")
        or record.get("debtors"))

    if record.get("turnover") is not None:
        return record

    # A holding company's debtors are usually just money owed by its own
    # subsidiaries. Nothing was sold, so five times that is not a turnover.
    if record.get("trade_debtors") is None and record.get("debtors_intercompany"):
        record["turnover_estimate_basis"] = (
            "no estimate: debtors are amounts owed by group undertakings, "
            "not trade debtors")
        return record

    hit = estimate.estimate_turnover(record.get("trade_debtors"),
                                     record.get("debtors"))
    if hit:
        record["turnover_estimate"] = hit["value"]
        record["turnover_estimate_basis"] = hit["working"]
        record["turnover_estimate_confidence"] = hit["confidence"]
        record["turnover_source"] = f"estimated from {hit['basis']}"
    return record


def _satisfied(found: Dict[str, Any], want_turnover: bool,
               saw_no_pl: bool) -> bool:
    if found["debtors"] is None or found["employees"] is None:
        return False
    # Trade debtors sit in the notes after the balance sheet and are what
    # the turnover estimate is built on. With one OCR worker the reader
    # checks after every page, and stopping as soon as total debtors were in
    # hand meant the debtors note was never read at all.
    if found.get("trade_debtors") is None and not found.get(
            "intercompany_debtors"):
        return False
    if found.get("net_assets") is None:
        return False
    if want_turnover and not saw_no_pl and found["turnover"] is None:
        return False
    return True


def read_filing_pdf(raw: bytes, want_turnover: bool = True,
                    allow_ocr: bool = True) -> Dict[str, Any]:
    """Read a PDF filing, stopping as soon as the figures are in hand."""
    state = {"found": None, "no_pl": False}

    def stop_when(pages):
        text = "\n".join(p["text"] for p in pages)
        if fields.no_profit_and_loss(text):
            state["no_pl"] = True
        state["found"] = extract_from_pages(pages, want_turnover)
        return _satisfied(state["found"], want_turnover, state["no_pl"])

    result = reader.read_pdf(raw, allow_ocr=allow_ocr, stop_when=stop_when)
    found = state["found"] or extract_from_pages(result["pages"], want_turnover)
    no_pl = state["no_pl"] or fields.no_profit_and_loss(result["text"])

    return {
        "found": found,
        "no_profit_and_loss": no_pl,
        "page_count": result["page_count"],
        "pages_read": result["pages_read"],
        "pages_ocred": result["pages_ocred"],
        "engine": result.get("engine"),
        "ocr_errors": result["ocr_errors"],
        "text": result["text"],
        "pages": result["pages"],
    }


def _flatten(found: Dict[str, Any]) -> Dict[str, Any]:
    """Values plus the evidence for each, so a wrong figure is traceable."""
    out: Dict[str, Any] = {}
    for key, target in (("turnover", "turnover"),
                        ("debtors", "debtors"),
                        ("trade_debtors", "trade_debtors"),
                        ("employees", "employee_size"),
                        ("net_assets", "net_assets"),
                        ("total_assets", "total_assets"),
                        ("total_liabilities", "total_liabilities")):
        hit = found.get(key)
        out[target] = hit["value"] if hit else None
        out[f"{target}_evidence"] = hit.get("evidence") if hit else None
        out[f"{target}_page"] = hit.get("page") if hit else None
        out[f"{target}_units"] = hit.get("units") if hit else None
        out[f"{target}_raw"] = hit.get("raw") if hit else None
        if hit and hit.get("working"):
            out[f"{target}_working"] = hit["working"]

    # Turnover and debtors are read off different pages, and a filing where
    # one page declares thousands and the other does not is either a
    # misdetected heading or a genuinely mixed presentation. Either way the
    # ratio between them is meaningless, so say so rather than letting it
    # flow into an estimate.
    inter = found.get("intercompany_debtors")
    out["debtors_intercompany"] = bool(
        inter and out.get("debtors") is not None
        and abs(inter["value"] - out["debtors"]) < 1)

    units = [out.get(f"{k}_units") for k in ("turnover", "debtors")
             if out.get(f"{k}_units")]
    out["scale_conflict"] = len(set(units)) > 1

    # pass, fail or not checked. A fail means a line on the balance sheet
    # page was misread or missed, so the row wants a look before it is used.
    check = found.get("balance_check") or {}
    out["balance_check"] = check.get("status", "not checked")
    out["balance_check_detail"] = check.get("summary")
    return out


def company(ch: ch_client.CompaniesHouse, number: str,
            name: Optional[str] = None, max_filings: int = MAX_FILINGS,
            allow_ocr: bool = True,
            keep_text: bool = False) -> Dict[str, Any]:
    number = ch_client.normalise(number)
    filings = ch.accounts_filings(number)

    if filings is None:
        return add_estimate({"company_number": number,
                             "status": "company_not_found", "turnover": None,
                             "debtors": None, "employee_size": None,
                             **EMPTY_BALANCE})
    if not filings:
        return add_estimate({"company_number": number,
                             "status": "no_accounts_filed", "turnover": None,
                             "debtors": None, "employee_size": None,
                             **EMPTY_BALANCE})

    attempts: List[Dict[str, Any]] = []

    for filing in filings[:max_filings]:
        meta_response = ch.get(filing["links"]["document_metadata"],
                               follow=False)
        if meta_response.status_code >= 400:
            attempts.append({"date": filing.get("date"),
                             "error": f"metadata {meta_response.status_code}"})
            continue

        meta = meta_response.json()
        resources = meta.get("resources") or {}
        doc_url = (meta.get("links") or {}).get("document")
        if not doc_url:
            attempts.append({"date": filing.get("date"),
                             "error": "no document link"})
            continue

        available = list(resources)
        atype = fields.accounts_type(filing.get("description", ""))
        want_turnover = fields.turnover_on_record(atype)

        base = {
            "company_number": number,
            "company_name": name,
            "accounts_type": atype,
            "turnover_on_record": want_turnover,
            "made_up_date": filing.get("action_date") or filing.get("date"),
            "filing_date": filing.get("date"),
            "available_resources": available,
        }

        # ---------------------------------------------------- iXBRL route
        zip_type = next((k for k in available if "zip" in k.lower()), None)
        xhtml_type = ("application/xhtml+xml"
                      if "application/xhtml+xml" in resources else None)

        if zip_type or xhtml_type:
            wanted = zip_type or xhtml_type
            response = ch.get(doc_url, accept=wanted)
            content = response.content if response.status_code < 400 else b""

            if content and zip_type:
                inner = ixbrl.ixbrl_from_zip(content)
                if inner is None:
                    attempts.append({
                        "date": filing.get("date"),
                        "error": "zip contained no iXBRL",
                        "zip_contents": ixbrl.zip_contents(content)[:10]})
                    content = b""
                else:
                    content = inner

            if content:
                try:
                    parsed = ixbrl.parse(content)
                except Exception as exc:
                    attempts.append({"date": filing.get("date"),
                                     "error": f"ixbrl parse: {exc}"[:150]})
                    parsed = None

                if parsed:
                    attempts.append({
                        "date": filing.get("date"),
                        "route": "zip-ixbrl" if zip_type else "ixbrl",
                        "facts": parsed["tagged_facts"],
                        "unparsed_facts": parsed["unparsed_facts"]})
                    if any(parsed[k] is not None for k in
                           ("turnover", "employee_size", "debtors",
                            "net_assets", "total_assets")):
                        return add_estimate({
                            **base,
                            "status": "ok",
                            "route": "zip-ixbrl" if zip_type else "ixbrl",
                            "turnover": parsed["turnover"],
                            "debtors": parsed["debtors"],
                            "trade_debtors": parsed["trade_debtors"],
                            "employee_size": parsed["employee_size"],
                            "turnover_tag": parsed["turnover_tag"],
                            "debtors_tag": parsed["debtors_tag"],
                            "trade_debtors_tag": parsed["trade_debtors_tag"],
                            "employee_tag": parsed["employee_tag"],
                            "net_assets": parsed["net_assets"],
                            "net_assets_tag": parsed["net_assets_tag"],
                            "total_assets": parsed["total_assets"],
                            "total_assets_working": parsed["total_assets_basis"],
                            "total_liabilities": parsed["total_liabilities"],
                            "total_liabilities_working":
                                parsed["total_liabilities_basis"],
                            "balance_sheet_date": parsed["balance_sheet_date"],
                            "debtors_intercompany":
                                parsed["debtors_intercompany"],
                            "balance_check": "not checked",
                            "balance_check_detail":
                                "iXBRL tagged figures, no OCR to misread",
                            "tagged_facts": parsed["tagged_facts"],
                            "attempts": attempts,
                        })
            elif response.status_code >= 400:
                attempts.append({"date": filing.get("date"),
                                 "error": f"{wanted} {response.status_code}"})

        # ------------------------------------------------------ PDF route
        if "application/pdf" not in resources and available:
            continue

        response = ch.get(doc_url, accept="application/pdf")
        if response.status_code >= 400:
            attempts.append({"date": filing.get("date"),
                             "error": f"pdf {response.status_code}"})
            continue

        try:
            result = read_filing_pdf(response.content, want_turnover,
                                     allow_ocr)
        except (ValueError, RuntimeError) as exc:
            attempts.append({"date": filing.get("date"),
                             "error": str(exc)[:150]})
            continue

        flat = _flatten(result["found"])
        attempts.append({
            "date": filing.get("date"), "route": "pdf",
            "engine": result.get("engine"),
            "pages": result["page_count"],
            "pages_read": result["pages_read"],
            "pages_ocred": result["pages_ocred"]})

        got_anything = any(flat[k] is not None for k in
                           ("turnover", "debtors", "employee_size",
                            "net_assets", "total_assets"))
        if got_anything:
            turnover_available = want_turnover and not result[
                "no_profit_and_loss"]
            return add_estimate({
                **base,
                "status": "ok",
                "route": "pdf",
                "turnover_on_record": turnover_available,
                "no_profit_and_loss": result["no_profit_and_loss"],
                "page_count": result["page_count"],
                "pages_read": result["pages_read"],
                "pages_ocred": result["pages_ocred"],
                "ocr_engine": result.get("engine"),
                "ocr_errors": result["ocr_errors"],
                "attempts": attempts,
                **flat,
                **({"text": result["text"]} if keep_text else {}),
            })

    return add_estimate({
        "company_number": number,
        "company_name": name,
        "status": "no_figures_found",
        "turnover": None,
        "debtors": None,
        "employee_size": None,
        **EMPTY_BALANCE,
        "attempts": attempts,
        "note": "small companies filing filleted accounts under section 444 "
                "do not deliver a profit and loss account, so turnover is "
                "not on the public record",
    })
