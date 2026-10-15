"""
The free route. When a filing carries iXBRL there is nothing to read and
nothing to guess: the figures are tagged.

On a real plc filing this returns turnover 372,104,000 and 2,006 employees
in about a second, against several minutes of OCR for the same document read
as a PDF. Always try this first.

Newer Companies House filings arrive as a zip report package with the iXBRL
inside, rather than as a bare xhtml file, so both shapes are handled.
"""

import io
import re
import zipfile
from typing import Any, Dict, List, Optional

from lxml import etree

# In tag priority order. Specific names first so a company that tags both
# TurnoverRevenue and a generic Revenue gets the specific one.
TURNOVER_TAGS = [
    "turnoverrevenue",
    "revenuefromcontractswithcustomers",
    "turnovergrossoperatingrevenue",
    "totalrevenue",
    "turnover",
    "revenuefromsaleofgoods",
    "revenuefromrenderingservices",
    "grossrevenue",
    "revenue",
]

EMPLOYEE_TAGS = [
    "averagenumberemployeesduringperiod",
    "averagenumberofemployeesincludingdirectors",
    "averagenumberofemployeesduringperiod",
    "averagenumberemployeestotal",
    "employeestotalaveragenumber",
    "numberofemployees",
]

DEBTOR_TAGS = [
    "debtors",
    "tradeandothercurrentreceivables",
    "tradeandotherreceivables",
    "totaldebtors",
    "tradedebtorstradereceivables",
    "tradereceivables",
]

# Trade debtors specifically. This is the balance that came from selling
# something, so it is the right basis for estimating turnover. The Debtors
# total above also carries other debtors, director loans and intercompany
# balances, which have nothing to do with sales.
TRADE_DEBTOR_TAGS = [
    "tradedebtorstradereceivables",
    "tradereceivables",
    "tradedebtors",
    "currenttradereceivables",
]


# Balance sheet. Matched exactly, never as a substring: "assets" is inside
# half the names in the taxonomy. FRS 102 (core:) and IFRS (ifrs-full:)
# names are both listed.
NET_ASSETS_TAGS = [
    "netassetsliabilities",
    "netassetsliabilitiesincludingpensionassetliability",
    "totalequity",
    "equity",
    "shareholdersfunds",
    "netassets",
]
TOTAL_ASSETS_TAGS = ["assets", "totalassets"]
TOTAL_LIABILITIES_TAGS = ["liabilities", "totalliabilities"]
FIXED_ASSETS_TAGS = ["fixedassets", "noncurrentassets", "totalfixedassets"]
CURRENT_ASSETS_TAGS = ["currentassets", "totalcurrentassets"]
TALCL_TAGS = ["totalassetslesscurrentliabilities"]
NET_CURRENT_TAGS = ["netcurrentassetsliabilities", "netcurrentassets"]
# Debtors that are only money owed by other group companies. Nothing was sold
# to anyone, so a turnover estimate built on them would be fiction.
INTERCOMPANY_DEBTOR_TAGS = ["amountsowedbygroupundertakings",
                            "amountsowedbygroupundertakingsparticipatinginterests",
                            "amountsowedbygroupundertakingsandundertakingsinwhichcompanyhasparticipatinginterest",
                            "amountsowedbyrelatedparties"]
CURRENT_LIABILITY_TAGS = ["currentliabilities", "creditorsduewithinoneyear",
                          "totalcurrentliabilities"]


def localname(el) -> str:
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag.split(":", 1)[-1]


def zip_contents(raw: bytes) -> List[str]:
    try:
        return zipfile.ZipFile(io.BytesIO(raw)).namelist()
    except Exception:
        return []


def ixbrl_from_zip(raw: bytes) -> Optional[bytes]:
    """Pull the tagged report out of a report package."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception:
        return None

    def rank(name: str) -> int:
        low = name.lower()
        if "/reports/" in low and low.endswith((".xhtml", ".xhtm", ".html")):
            return 0
        if low.endswith((".xhtml", ".xhtm")):
            return 1
        if low.endswith((".html", ".htm")):
            return 2
        if low.endswith(".xml") and not any(
                k in low for k in ("manifest", "meta", "catalog", "package",
                                   "_cal", "_def", "_lab", "_pre")):
            return 3
        return 9

    candidates = sorted([n for n in zf.namelist() if rank(n) < 9], key=rank)
    for name in candidates:
        try:
            data = zf.read(name)
        except Exception:
            continue
        if b"nonFraction" in data or b"nonfraction" in data or b"ix:" in data:
            return data
    return zf.read(candidates[0]) if candidates else None


def _contexts(root) -> Dict[str, Dict[str, Any]]:
    """Period end and whether the context carries dimensions.

    Dimensions matter. A segmental revenue figure is tagged with the same
    name as the group total but sits in a context with an explicit member.
    Preferring undimensioned contexts is what keeps a single division's
    revenue from being reported as the whole company's.
    """
    out = {}
    for el in root.iter():
        if localname(el).lower() != "context":
            continue
        cid = el.get("id")
        if not cid:
            continue
        end, dimensioned = "", False
        for child in el.iter():
            name = localname(child).lower()
            if name in ("instant", "enddate") and child.text:
                end = child.text.strip()
            if name in ("explicitmember", "typedmember"):
                dimensioned = True
        out[cid] = {"end": end, "dimensioned": dimensioned}
    return out


def _value(el) -> Optional[float]:
    text = "".join(el.itertext()).strip()
    if not text:
        return None
    negative = "(" in text and ")" in text
    cleaned = re.sub(r"[^\d.]", "", text)
    # a stray full stop with three digits after it is a thousands separator
    cleaned = re.sub(r"\.(?=\d{3}(?:\D|$))", "", cleaned)
    if not cleaned or cleaned == ".":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    scale = el.get("scale")
    if scale:
        try:
            value *= 10 ** int(scale)
        except ValueError:
            pass
    if el.get("sign") == "-" or negative:
        value = -value
    return value


def parse(raw: bytes) -> Dict[str, Any]:
    parser = etree.HTMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(raw, parser)
    if root is None:
        raise ValueError("could not parse iXBRL document")

    contexts = _contexts(root)
    facts, unparsed = [], 0
    for el in root.iter():
        if localname(el).lower() != "nonfraction":
            continue
        value = _value(el)
        if value is None:
            unparsed += 1
            continue
        name = el.get("name") or ""
        ref = el.get("contextRef") or el.get("contextref") or ""
        ctx = contexts.get(ref, {})
        facts.append({
            "name": name,
            "short": name.split(":")[-1].lower(),
            "value": value,
            "period_end": ctx.get("end", ""),
            "dimensioned": ctx.get("dimensioned", False),
        })

    def pick(tags, positive_only=True):
        for tag in tags:
            hits = [f for f in facts if f["short"] == tag]
            if not hits:
                hits = [f for f in facts if tag in f["short"]]
            if positive_only:
                hits = [h for h in hits if h["value"] > 0]
            if not hits:
                continue
            # group totals first, then the most recent period
            plain = [h for h in hits if not h["dimensioned"]] or hits
            plain.sort(key=lambda f: (f["period_end"] or "", abs(f["value"])),
                       reverse=True)
            return plain[0]
        return None

    def exact(tags, period=None):
        """Undimensioned fact with exactly this name, latest period, or the
        given period so every balance sheet figure comes from one date."""
        for tag in tags:
            hits = [f for f in facts if f["short"] == tag
                    and not f["dimensioned"]]
            if period:
                hits = [h for h in hits if h["period_end"] == period]
            if hits:
                hits.sort(key=lambda f: f["period_end"] or "", reverse=True)
                return hits[0]
        return None

    net = exact(NET_ASSETS_TAGS)
    period = net["period_end"] if net else None
    total_assets = exact(TOTAL_ASSETS_TAGS, period)
    assets_basis = total_assets["name"] if total_assets else None
    if total_assets is None:
        # Fixed assets are often tagged only by type (FixedAssetInvestments,
        # PropertyPlantEquipment...) with no FixedAssets total, so adding
        # fixed + current misses them. Work from the lines every statutory
        # balance sheet carries instead:
        #   creditors due within one year = current assets - net current assets
        #   total assets = total assets less current liabilities + those creditors
        fixed = exact(FIXED_ASSETS_TAGS, period)
        current = exact(CURRENT_ASSETS_TAGS, period)
        net_current = exact(NET_CURRENT_TAGS, period)
        talcl = exact(TALCL_TAGS, period)
        cl = exact(CURRENT_LIABILITY_TAGS, period)
        if talcl is not None and current is not None and net_current is not None:
            creditors = current["value"] - net_current["value"]
            value = talcl["value"] + creditors
            total_assets = {"value": value, "period_end": talcl["period_end"]}
            assets_basis = (f"total assets less current liabilities "
                            f"{talcl['value']:,.0f} + creditors within one year "
                            f"{creditors:,.0f} (current assets "
                            f"{current['value']:,.0f} - net current assets "
                            f"{net_current['value']:,.0f})")
        elif talcl is not None and cl is not None:
            value = talcl["value"] + abs(cl["value"])
            total_assets = {"value": value, "period_end": talcl["period_end"]}
            assets_basis = "total assets less current liabilities + " \
                           "current liabilities"
        elif current is not None and fixed is not None:
            value = current["value"] + fixed["value"]
            total_assets = {"value": value, "period_end": current["period_end"]}
            assets_basis = "fixed assets + current assets"
    total_liabilities = exact(TOTAL_LIABILITIES_TAGS, period)
    liabilities_basis = total_liabilities["name"] if total_liabilities else None
    if total_liabilities is not None:
        total_liabilities = {**total_liabilities,
                             "value": abs(total_liabilities["value"])}
    elif total_assets is not None and net is not None:
        value = total_assets["value"] - net["value"]
        if value >= 0:
            total_liabilities = {"value": value}
            liabilities_basis = "total assets - net assets"

    turnover = pick(TURNOVER_TAGS)
    debtors_hit = pick(DEBTOR_TAGS)
    owed_by_group = None
    if debtors_hit is not None:
        for f in facts:
            if f["short"] in INTERCOMPANY_DEBTOR_TAGS and \
                    f["period_end"] == debtors_hit["period_end"]:
                owed_by_group = (owed_by_group or 0) + f["value"]
    employees = pick(EMPLOYEE_TAGS)
    debtors = pick(DEBTOR_TAGS)
    trade_debtors = pick(TRADE_DEBTOR_TAGS)

    return {
        "turnover": turnover["value"] if turnover else None,
        "turnover_tag": turnover["name"] if turnover else None,
        "turnover_period_end": turnover["period_end"] if turnover else None,
        "employee_size": int(employees["value"]) if employees else None,
        "employee_tag": employees["name"] if employees else None,
        "debtors": debtors["value"] if debtors else None,
        "debtors_tag": debtors["name"] if debtors else None,
        "trade_debtors": trade_debtors["value"] if trade_debtors else None,
        "trade_debtors_tag": trade_debtors["name"] if trade_debtors else None,
        "debtors_intercompany": bool(
            debtors_hit and owed_by_group is not None
            and abs(owed_by_group - debtors_hit["value"]) < 1),
        "net_assets": net["value"] if net else None,
        "net_assets_tag": net["name"] if net else None,
        "total_assets": total_assets["value"] if total_assets else None,
        "total_assets_basis": assets_basis,
        "total_liabilities": total_liabilities["value"]
        if total_liabilities else None,
        "total_liabilities_basis": liabilities_basis,
        "balance_sheet_date": period,
        "tagged_facts": len(facts),
        "unparsed_facts": unparsed,
    }