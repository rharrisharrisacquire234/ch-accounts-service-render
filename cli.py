"""
One tool for everything you used to need three for.

    python cli.py engine                         check the OCR engine works
    python cli.py pages accounts.pdf --page 3    export a page image to test by hand
    python cli.py parse page3.md                 run model output through the rules
    python cli.py file accounts.pdf              read a PDF you already have
    python cli.py file accounts.pdf --engine unlimited   read it with the VLM
    python cli.py compare accounts.pdf           read it both ways, side by side
    python cli.py company 09760850               fetch and read one company
    python cli.py company 09760850 --show        also print what it read
    python cli.py batch numbers.txt --workers 4  a whole list
    python cli.py dump 09760850                  every iXBRL fact to csv

The point of `file` is that it needs no API key and no rate limit, so it is
the right way to tune settings against a document you already trust.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import time
from collections import Counter

import ch_client
import estimate
import figures
import fields
import ixbrl
import markdown_rows
import pipeline
import reader


def _ch():
    key = os.getenv("CH_API_KEY", "")
    if not key:
        sys.exit("CH_API_KEY is not set.\n"
                 '  PowerShell:  $env:CH_API_KEY = "your key"\n'
                 "  bash:        export CH_API_KEY=your_key")
    return ch_client.CompaniesHouse(key)


def _figure(value):
    if value is None:
        return "not found"
    return f"{value:,.0f}"


def _report(found, no_pl, record=None):
    for label, key in (("turnover", "turnover"), ("debtors", "debtors"),
                       ("trade debtors", "trade_debtors"),
                       ("employees", "employees"),
                       ("net assets", "net_assets"),
                       ("total assets", "total_assets"),
                       ("liabilities", "total_liabilities")):
        hit = found.get(key)
        if hit:
            page = hit.get("page")
            print(f"  {label:<13} {_figure(hit['value']):>15}   page {page}")
            print(f"  {'':<13} from: {(hit.get('working') or hit.get('evidence') or '')[:88]}")
        elif label == "turnover" and no_pl:
            print(f"  {label:<13} {'not filed':>15}   "
                  "profit and loss withheld under s444")
        else:
            print(f"  {label:<13} {'not found':>15}")

    if record and record.get("turnover_estimate") is not None:
        print()
        print(f"  {'estimate':<13} {record['turnover_estimate']:>15,.0f}   "
              f"{record['turnover_estimate_confidence']} confidence")
        print(f"  {'':<13} {record['turnover_estimate_basis']}")
        print(f"  {'':<13} an estimate, not a filed figure")


def cmd_engine(args):
    """Smoke test before you spend an hour of batch time finding out."""
    if args.engine:
        reader.set_engine(args.engine)
    print(f"engine       {reader.OCR_ENGINE}")
    print(f"scout engine {os.getenv('SCOUT_ENGINE', reader.OCR_ENGINE)}")
    print(f"read dpi     {reader.READ_DPI}")
    print(f"repair mode  {figures.REPAIR_MODE}")
    if reader.OCR_ENGINE == "unlimited":
        import unlimited_client
        print(f"mode         {unlimited_client.UNLIMITED_MODE}")
        if unlimited_client.UNLIMITED_MODE.lower() == "local":
            report = unlimited_client.torch_report()
            print(f"torch        {report['torch']}, "
                  f"cuda build {report['cuda_build'] or 'none'}")
            print(f"gpu          "
                  f"{', '.join(report['devices']) or 'none available'}")
            if report["vram_total_gb"]:
                print(f"vram         {report['vram_free_gb']}GB free of "
                      f"{report['vram_total_gb']}GB, need about "
                      f"{unlimited_client.VRAM_WANTED_GB:g}GB")
                print(f"capability   {report['capability']}, bfloat16 "
                      f"{'yes' if report['bf16'] else 'no, using float16'}")
            if report["note"]:
                print(f"             {report['note']}")
            print(f"device       {unlimited_client.pick_device()}"
                  f"{' 4bit' if unlimited_client.UNLIMITED_4BIT else ''}")
        else:
            print(f"url          {unlimited_client.UNLIMITED_URL}")
    try:
        print(f"status       ok, {reader.check_ocr()}")
    except RuntimeError as exc:
        print(f"status       FAILED\n\n{exc}")
        return 1

    if args.page:
        raw = open(args.page, "rb").read()
        import pymupdf
        from layout import page_image
        pdf = pymupdf.open(stream=raw, filetype="pdf")
        try:
            image = page_image(pdf.load_page(args.page_index), reader.READ_DPI)
        finally:
            pdf.close()
        t0 = time.time()
        index, rows, markdown, err = reader.read_engine().run(
            [(args.page_index, image)])[0]
        if err:
            print(f"\npage {index} failed: {err}")
            return 1
        print(f"\npage {index} read in {time.time() - t0:.1f}s, "
              f"{len(rows)} rows\n")
        print("=" * 70)
        print(markdown)
    return 0


def cmd_pages(args):
    """Export page images so a page can be dropped into a hosted demo.

    The question worth answering first is whether the model reads a
    Companies House scan better than Tesseract does. That does not need a
    GPU, a server or an install. It needs one page as a PNG and five
    minutes on the Hugging Face demo at
    https://huggingface.co/spaces/baidu/Unlimited-OCR
    """
    import pymupdf
    from layout import page_image
    raw = open(args.path, "rb").read()
    pdf = pymupdf.open(stream=raw, filetype="pdf")
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.path))[0]
    try:
        wanted = range(pdf.page_count) if args.all else \
            [p for p in args.page or [0] if 0 <= p < pdf.page_count]
        for index in wanted:
            image = page_image(pdf.load_page(index), args.dpi)
            path = os.path.join(args.out, f"{base}_p{index:03d}.png")
            image.save(path)
            print(f"{path}  {image.width}x{image.height}")
    finally:
        pdf.close()
    print(f"\nDrop one of these into "
          f"https://huggingface.co/spaces/baidu/Unlimited-OCR\n"
          f"and compare what comes back with: python cli.py file "
          f"{args.path} --show")


def cmd_parse(args):
    """Run a saved markdown file through the extraction rules.

    Paste what the model returned into a .txt or .md file and point this at
    it. No GPU, no server, no install. It answers the question that actually
    matters: would the pipeline read the right figures out of that output.
    """
    raw = open(args.path, encoding="utf-8", errors="ignore").read()
    figures.set_repair_mode(args.repair)
    rows = markdown_rows.rows_from_markdown(raw)
    text = markdown_rows.markdown_text(raw)
    pages = [{"index": args.page, "rows": rows, "text": text}]

    print(f"{args.path}, {len(raw):,} chars, {len(rows)} rows, "
          f"repair {args.repair}\n")

    years = fields.year_columns(rows)
    current = fields.current_year_column(years) if years else None
    if years:
        print("  year columns   " + ", ".join(
            f"{y} at x{int(x)}" for y, x in sorted(years.items())))
        print(f"  current year   {current[0] if current else 'none'}")
    else:
        print("  year columns   none found, will fall back to leftmost")
    print()

    found = {
        "turnover": fields.find_turnover(pages),
        "debtors": fields.find_debtors(pages),
        "trade debtors": fields.find_trade_debtors(pages),
        "employees": fields.find_employees(pages),
    }
    for name, hit in found.items():
        if not hit:
            print(f"  {name:<15} not found")
            continue
        year = f"  ({hit.get('year')})" if hit.get("year") else ""
        print(f"  {name:<15} {_figure(hit['value']):>14}{year}")
        print(f"                  from: {hit.get('evidence', '')[:90]}")

    if args.show:
        print("\n" + "=" * 70)
        for row in rows:
            cells = " ".join(f"[{c.value:g}@{int(c.x)}]" for c in row.cells)
            print(f"{row.text[:70]:<72}{cells}")
    return 0


def cmd_compare(args):
    """The same PDF through both engines, so the difference is a fact rather
    than an opinion. Run this on a filing you already know the answer to."""
    raw = open(args.path, "rb").read()
    print(f"{args.path}, {len(raw):,} bytes\n")
    table = []
    for engine in ("tesseract", "unlimited"):
        reader.set_engine(engine)
        t0 = time.time()
        try:
            result = pipeline.read_filing_pdf(raw)
        except Exception as exc:
            print(f"{engine:<12} failed: {str(exc).splitlines()[0]}")
            continue
        flat = pipeline._flatten(result["found"])
        flat["engine"] = engine
        flat["secs"] = round(time.time() - t0, 1)
        flat["pages_ocred"] = result["pages_ocred"]
        table.append(flat)
        print(f"{engine}  {result['pages_ocred']} pages, "
              f"{flat['secs']}s")
        _report(result["found"], result["no_profit_and_loss"])
        print()

    if len(table) == 2:
        print("=" * 70)
        for key in ("turnover", "debtors", "trade_debtors", "employee_size"):
            a, b = table[0][key], table[1][key]
            mark = "same" if a == b else "DIFFERENT"
            print(f"  {key:<15} tesseract={_figure(a):>14}  "
                  f"unlimited={_figure(b):>14}   {mark}")


def cmd_file(args):
    if args.engine:
        reader.set_engine(args.engine)
    raw = open(args.path, "rb").read()
    print(f"{args.path}, {len(raw):,} bytes")
    t0 = time.time()
    result = pipeline.read_filing_pdf(raw, want_turnover=not args.no_turnover)
    print(f"{result['page_count']} pages, read {result['pages_read']}, "
          f"{result['pages_ocred']} {result.get('engine')} calls, "
          f"{time.time() - t0:.1f}s")
    if result["ocr_errors"]:
        print(f"ocr errors: {result['ocr_errors']}")
    print()
    record = pipeline.add_estimate(pipeline._flatten(result["found"]))
    _report(result["found"], result["no_profit_and_loss"], record)
    if args.show:
        print("\n" + "=" * 70)
        print(result["text"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result["text"])
        print(f"\ntext written to {args.out}")


def cmd_company(args):
    if args.engine:
        reader.set_engine(args.engine)
    ch = _ch()
    t0 = time.time()
    result = pipeline.company(ch, args.number, max_filings=args.max_filings,
                              allow_ocr=not args.no_ocr,
                              keep_text=args.show)
    print(json.dumps({k: v for k, v in result.items() if k != "text"},
                     indent=2, default=str))
    print(f"\n{time.time() - t0:.1f}s")
    if args.show and result.get("text"):
        print("=" * 70)
        print(result["text"])


def cmd_batch(args):
    if args.engine:
        reader.set_engine(args.engine)
    ch = _ch()
    numbers = []
    # The first argument is a file if one exists at that path, otherwise it
    # is just another company number. Saves you having to make a one line
    # text file to check three companies.
    candidates = ([args.path] if args.path else []) + list(args.numbers)
    for item in candidates:
        if os.path.isfile(item):
            with open(item) as fh:
                numbers += [line.strip() for line in fh
                            if line.strip() and not line.startswith("#")]
        else:
            numbers.append(item)
    if not numbers:
        sys.exit("give me some company numbers, or a file with one per line")

    routes, statuses = Counter(), Counter()
    hits = Counter()
    possible = 0
    ocr_pages = 0
    estimated = 0
    observed = []
    t0 = time.time()
    rows = []

    def run(number):
        start = time.time()
        try:
            out = pipeline.company(ch, number, allow_ocr=not args.no_ocr)
        except Exception as exc:
            out = {"company_number": number, "status": f"error: {exc}"}
        out["secs"] = round(time.time() - start, 1)
        return out

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(run, numbers):
            rows.append(out)
            routes[out.get("route") or "none"] += 1
            statuses[out.get("status")] += 1
            ocr_pages += out.get("pages_ocred") or 0
            if out.get("turnover_on_record"):
                possible += 1
            for key in ("turnover", "debtors", "employee_size"):
                if out.get(key) is not None:
                    hits[key] += 1
            if out.get("observed_debtor_multiple"):
                observed.append(out["observed_debtor_multiple"])
            shown = out.get("turnover")
            mark = " "
            if shown is None and out.get("turnover_estimate") is not None:
                shown = out["turnover_estimate"]
                mark = "~"          # estimated, not filed
                estimated += 1
            print(f"{out['company_number']:>10}  "
                  f"{str(out.get('route')):9}  "
                  f"t={mark}{_figure(shown):>13}  "
                  f"d={_figure(out.get('debtors')):>13}  "
                  f"e={str(out.get('employee_size') or '-'):>6}  "
                  f"ocr={out.get('pages_ocred') or 0:>3}  "
                  f"{out['secs']:>5}s  {out.get('status')}")

    total = len(numbers)
    elapsed = time.time() - t0
    print("\n" + "=" * 74)
    print(f"companies    {total} in {elapsed:.0f}s ({elapsed / total:.1f}s each)")
    print(f"routes       {dict(routes)}")
    print(f"statuses     {dict(statuses)}")
    print(f"turnover     {hits['turnover']}/{total} found, "
          f"{possible}/{total} could have it on record")
    print(f"debtors      {hits['debtors']}/{total}")
    print(f"employees    {hits['employee_size']}/{total}")
    print(f"estimated    {estimated}/{total} turnover figures marked ~ "
          f"are estimates, not filed")
    print(f"pages OCRed  {ocr_pages}")
    if observed:
        observed.sort()
        median = observed[len(observed) // 2]
        print(f"\nfor information, the turnover to debtors ratio actually "
              f"observed on the\n{len(observed)} companies that filed both "
              f"figures and look internally consistent:")
        print(f"  min {min(observed):.1f}   median {median:.1f}   "
              f"max {max(observed):.1f}")
        print(f"  the estimator uses a fixed {estimate.MULTIPLE:g}, set in "
              f"estimate.py")
        if len(observed) < 20:
            print("  too few to draw a conclusion from yet")
    print(f"\nprojected for 5,000 a day at {args.workers} workers: "
          f"{elapsed / total * 5000 / 3600:.1f} hours")

    if args.csv:
        keys = ["company_number", "status", "route", "accounts_type",
                "filing_date", "turnover", "turnover_source",
                "turnover_estimate", "turnover_estimate_confidence",
                "turnover_estimate_basis", "observed_debtor_multiple",
                "debtors", "trade_debtors", "employee_size",
                "net_assets", "total_assets", "total_liabilities",
                "turnover_on_record", "pages_ocred", "secs"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"written to {args.csv}")


def cmd_dump(args):
    ch = _ch()
    number = ch_client.normalise(args.number)
    filings = ch.accounts_filings(number)
    if not filings:
        sys.exit("no accounts filings")

    for i, filing in enumerate(filings[:6]):
        print(f"  [{i}] {filing.get('date')}  {filing.get('description')}")
    filing = filings[args.filing]
    print(f"\nusing [{args.filing}] {filing.get('date')}")

    meta = ch.get(filing["links"]["document_metadata"], follow=False).json()
    resources = meta.get("resources") or {}
    print(f"resources: {list(resources)}")

    zip_type = next((k for k in resources if "zip" in k.lower()), None)
    xhtml = "application/xhtml+xml" if "application/xhtml+xml" in resources \
        else None
    if not (zip_type or xhtml):
        sys.exit("no iXBRL on this filing, use `cli.py company` instead")

    response = ch.get(meta["links"]["document"], accept=zip_type or xhtml)
    content = response.content
    if zip_type:
        print(f"zip contents: {ixbrl.zip_contents(content)}")
        content = ixbrl.ixbrl_from_zip(content)
        if content is None:
            sys.exit("zip contained no iXBRL")

    from lxml import etree
    root = etree.fromstring(content, etree.HTMLParser(recover=True,
                                                      huge_tree=True))
    contexts = ixbrl._contexts(root)
    rows = []
    for el in root.iter():
        if ixbrl.localname(el).lower() != "nonfraction":
            continue
        name = el.get("name") or ""
        ref = el.get("contextRef") or el.get("contextref") or ""
        ctx = contexts.get(ref, {})
        rows.append({"tag": name.split(":")[-1], "full_name": name,
                     "value": ixbrl._value(el),
                     "period_end": ctx.get("end", ""),
                     "dimensioned": ctx.get("dimensioned", False)})

    path = f"facts_{number}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "full_name", "value",
                                                "period_end", "dimensioned"])
        writer.writeheader()
        writer.writerows(rows)
    unparsed = sum(1 for r in rows if r["value"] is None)
    print(f"{len(rows)} tagged facts ({unparsed} with an unreadable value) "
          f"written to {path}")

    print("\nparser returns:")
    print(json.dumps(ixbrl.parse(content), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("engine", help="check the OCR engine is reachable")
    p.add_argument("--engine", choices=["tesseract", "unlimited"])
    p.add_argument("--page", help="optional PDF to read one page of")
    p.add_argument("--page-index", type=int, default=0)
    p.set_defaults(func=cmd_engine)

    p = subs.add_parser("pages", help="export page images for manual testing")
    p.add_argument("path")
    p.add_argument("--page", type=int, nargs="*", default=[0])
    p.add_argument("--all", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out", default="pages")
    p.set_defaults(func=cmd_pages)

    p = subs.add_parser("parse", help="run saved model markdown through the rules")
    p.add_argument("path")
    p.add_argument("--page", type=int, default=0)
    p.add_argument("--repair", choices=["light", "full"], default="light")
    p.add_argument("--show", action="store_true",
                   help="print every row with the figures it found")
    p.set_defaults(func=cmd_parse)

    p = subs.add_parser("compare", help="read one PDF with both engines")
    p.add_argument("path")
    p.set_defaults(func=cmd_compare)

    p = subs.add_parser("file", help="read a PDF from disk")
    p.add_argument("path")
    p.add_argument("--engine", choices=["tesseract", "unlimited"],
                   help="override OCR_ENGINE for this run")
    p.add_argument("--show", action="store_true", help="print the text read")
    p.add_argument("--out", help="write the text to a file")
    p.add_argument("--no-turnover", action="store_true",
                   help="skip turnover, for filleted accounts")
    p.set_defaults(func=cmd_file)

    p = subs.add_parser("company", help="fetch and read one company")
    p.add_argument("number")
    p.add_argument("--max-filings", type=int, default=3)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--engine", choices=["tesseract", "unlimited"])
    p.set_defaults(func=cmd_company)

    p = subs.add_parser("batch", help="run a list of company numbers")
    p.add_argument("path", nargs="?",
                   help="a company number, or a file with one per line")
    p.add_argument("numbers", nargs="*", help="more company numbers")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--csv", help="write results to a csv")
    p.add_argument("--engine", choices=["tesseract", "unlimited"])
    p.set_defaults(func=cmd_batch)

    p = subs.add_parser("dump", help="dump every iXBRL fact to csv")
    p.add_argument("number")
    p.add_argument("--filing", type=int, default=0)
    p.set_defaults(func=cmd_dump)

    args = parser.parse_args()
    if getattr(args, "engine", None):
        reader.set_engine(args.engine)
    if args.command not in ("engine", "compare", "dump", "pages", "parse"):
        try:
            reader.check_ocr()
        except RuntimeError as exc:
            print(f"warning: {exc}\n")
    args.func(args)


if __name__ == "__main__":
    main()