# Companies House accounts extractor

Pulls turnover, debtors, employee numbers and the balance sheet (net assets,
total assets, total liabilities) out of UK filed accounts.

Deploying on the Hostinger VPS and wiring it to n8n: see `DEPLOY.md`.

Every figure comes back with the line it was read from and the page it came
from, so a wrong number can be traced rather than argued about.

## What changed and why

The previous version read each page as a flat string and then hunted for a
number after a keyword. That loses the horizontal position of every figure,
and in a set of accounts the horizontal position *is* the meaning. On a real
balance sheet the flat text reads:

```
Debtors 6 493,712 517,202
```

A regex looking for a number after "Debtors" finds the note reference 6, or
picks up 2025 from the column heading two rows above. On the sample filing
in `samples/` the old code returned a debtors figure of **2025**.

This version keeps the coordinates, finds the year heading row, and works
out which column each figure belongs to. Same document, same OCR engine,
correct answer.

Four other things it now gets right:

**A full stop is not a decimal point.** At 200 dpi bitonal, Tesseract reads
`312,365` as `312.365` regularly. UK statutory accounts do not show pence,
so a full stop with exactly three digits after it is always a misread comma.
Without that rule you are out by a factor of a thousand and nothing warns you.

**Turnover is often not there at all.** A small company filing filleted
accounts under section 444 does not deliver a profit and loss account to the
registrar. The word Turnover still appears, three times, in the accounting
policy note, with no figures near it. The service now reports "not on the
public record" instead of returning whatever number happened to be nearby.

**Page selection.** A 195 page group filing keeps the income statement
around page 123. A fixed "first 15 pages" window never reaches it. This
version reads the contents page, scouts cheaply for the pages that mention
the statements, then reads only those properly, and stops as soon as it has
what it came for.

**Resources.** PDFs get closed, the rate limit retry is a bounded loop
rather than unbounded recursion, and the rate limit state is shared rather
than duplicated across uvicorn workers.

## Files

| File | What it does |
| --- | --- |
| `figures.py` | Turns an OCR token into a number, repairing the predictable misreads |
| `layout.py` | Rebuilds rows and columns from Tesseract word coordinates |
| `markdown_rows.py` | Rebuilds the same rows from Unlimited-OCR markdown |
| `ocr_engine.py` | One interface over both engines, so nothing else has to choose |
| `unlimited_client.py` | Talks to baidu/Unlimited-OCR, over http or in process |
| `fields.py` | Finds turnover, debtors and headcount in the row model |
| `checks.py` | Checks the balance sheet adds up, so a misread line is flagged |
| `reader.py` | Decides which pages to read and how hard |
| `ixbrl.py` | The free route: tagged figures, no OCR |
| `ch_client.py` | Companies House client, rate limit aware |
| `pipeline.py` | Picks the cheapest route per filing |
| `app.py` | The HTTP service |
| `cli.py` | Command line tool for files, companies and batches |
| `test_extract.py` | Pins every bug listed above so they cannot come back |

## Choosing an OCR engine

Two engines, same output. Set `OCR_ENGINE` and nothing else in the codebase
changes.

| | `tesseract` | `unlimited` |
| --- | --- | --- |
| What it is | classic OCR, word boxes | baidu/Unlimited-OCR, a vision language model |
| Hardware | any CPU | NVIDIA GPU, about 8GB, local or remote |
| Reads a table as | words with x positions | a markdown table with the columns intact |
| Cost per page | roughly a second | GPU seconds, depends on the card |
| Best at | clean single column filings | multi column annual reports, poor scans, odd layouts |

The reason both are kept is that most of your volume never reaches OCR at
all. iXBRL filings are exact and free, and a PDF with a text layer is free
too. OCR is the fallback, so paying for a GPU only makes sense if it fixes
filings Tesseract gets wrong. Measure that before you commit to it.

### Why the swap is safe

Everything downstream of `reader.py` works on a row model: tokens with an x
position, so the year columns can be told apart. Tesseract produces that
from word boxes. Unlimited-OCR does not return word boxes, it returns a
parsed document, so `markdown_rows.py` reads the structure instead. A table
row like

```
| Debtors | 6 | 493,712 | 517,202 |
```

already says which column each figure is in, which is the thing the
Tesseract path spends coordinate maths working out. Each cell gets a
synthetic x of its column index, and the existing column picker, the
separator repairs, the section 444 rules and every test carry over
unchanged. If the model ever returns a table whose header row has a
different number of cells to its body, the picker finds nothing in range and
falls back to the leftmost figure, which is the correct default anyway
because UK accounts always present the current year first.

`FIGURE_REPAIR` follows the engine. Tesseract needs the letter to digit
repairs, because at 200 dpi bitonal it reads `L00` for `100`. A vision model
reads characters in context and does not make that mistake, so those
substitutions are switched off and only the separator repair stays on. Set
`FIGURE_REPAIR=full` or `light` to override.

### Running the model

The model needs a GPU. There is no CPU path that finishes in a useful time,
so on a laptop without an NVIDIA card, leave `OCR_ENGINE=tesseract` and
point `UNLIMITED_URL` at a box that has one.

**Server route, the one to use.** The model sits behind vLLM or SGLang and
speaks the OpenAI chat API. This service then needs nothing but httpx.

```powershell
docker run --gpus all -p 10000:8000 --ipc=host `
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface `
  vllm/vllm-openai:unlimited-ocr `
  --model baidu/Unlimited-OCR --served-model-name Unlimited-OCR `
  --trust-remote-code --max-model-len 32768
```

```powershell
$env:OCR_ENGINE = "unlimited"
$env:UNLIMITED_URL = "http://your-gpu-box:10000"
$env:UNLIMITED_SERVER = "vllm"
python cli.py engine
```

For SGLang instead, follow the model card and set
`$env:UNLIMITED_SERVER = "sglang"`, which adds the repeat blocker and the
image mode the card specifies.

**Local route.** transformers loads the weights into this process. Simple,
but it puts torch and 8GB of VRAM inside the web service and serialises
requests. Use it for a single machine test, not for a batch.

Do this inside a virtual environment. It moves torch, transformers and
pillow, and anything else on the machine that depends on them will notice.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_gpu.txt

# the line that decides whether the GPU gets used at all
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

$env:OCR_ENGINE = "unlimited"
$env:UNLIMITED_MODE = "local"
python cli.py engine
```

`pip install torch` on its own gives the CPU only build, and the error it
produces later says "Torch not compiled with CUDA enabled", which does not
mention the missing index url. `python cli.py engine` now prints the torch
version, the CUDA build and the card it can see, so that failure is visible
in one line before any weights load.

If pip stops with `WinError 206: The filename or extension is too long`,
that is Windows refusing paths over 260 characters and torch shipping deeply
nested licence files. Enable long paths once in an admin PowerShell:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

or put the venv somewhere short like `C:\ocrenv`. The install stops
partway when this happens, so packages listed after torch never arrive and
the failure surfaces much later as a missing module.

`UNLIMITED_DEVICE` is `auto`, which uses a GPU if there is one and falls
back to the CPU if not. On a card too small to hold the model, `auto` still
picks `cuda` and then fails the preflight with the numbers, so set
`UNLIMITED_DEVICE=cpu` explicitly if you want the slow path. The CPU path works, in the sense that it returns
correct markdown, but it takes minutes per page. It is there so you can look
at one page and judge the quality, not so you can run a batch.

### Testing without any of this

The question worth answering first is whether the model reads a Companies
House scan better than Tesseract does, and that needs no GPU and no install:

```powershell
python cli.py pages samples\SC284050_aa_2026-01-27.pdf --page 3
```

Drop the PNG into https://huggingface.co/spaces/baidu/Unlimited-OCR and
compare what comes back with `python cli.py file <the same pdf> --show`. If
the model does not read the balance sheet noticeably better, nothing further
is worth paying for.

### What the card has to have

The weights are 6.7GB at bfloat16, so about 10GB of free VRAM is wanted once
the vision encoder and the KV cache are accounted for. `python cli.py engine`
now prints free and total VRAM and the compute capability before it loads
anything, because finding out after a 6.7GB download is a poor way to learn
it.

Two things that catch consumer cards. Anything below compute capability 8.0,
which is anything before the 30 series, has no bfloat16, so `UNLIMITED_DTYPE`
falls back to float16 automatically. And a 4GB or 6GB card cannot hold the
model at all. `UNLIMITED_4BIT=1` will quantise it to roughly 2GB using
bitsandbytes, but that quantises the vision encoder too, which is precisely
the part being judged, so treat any figure it produces as indicative and
never as a benchmark.

### Prove it before you trust it

```powershell
python cli.py engine --page samples\SC284050_aa_2026-01-27.pdf --page-index 3
python cli.py compare samples\SC284050_aa_2026-01-27.pdf
```

`engine` checks the model answers and prints the raw markdown for one page,
which is the fastest way to see whether it read the table as a table.
`compare` reads the same PDF with both engines and prints the figures side
by side, so a difference is a fact rather than an argument. Run it on
twenty filings you already know the answers to before switching a batch.

### Scouting

On a filing over 24 pages the reader takes a cheap look at pages to find
where the statements are, then reads only those properly. With Tesseract a
scout page costs about a second. With a vision model it costs nearly as much
as a real read, so scouting with the model can double the bill on a 195 page
group filing. Keep the cheap search on the CPU:

```powershell
$env:OCR_ENGINE = "unlimited"
$env:SCOUT_ENGINE = "tesseract"
```

Small filings are read whole and never scout, so this changes nothing for a
ten page set of small company accounts.

## Setting up on Windows

From `B:\Harris Acquire\OCR Model Test`, in PowerShell.

```powershell
# once
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tesseract, if you have not already
winget install UB-Mannheim.TesseractOCR
```

If Tesseract is not on PATH, point at it directly:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Then your Companies House key. `export` is bash and does nothing here:

```powershell
$env:CH_API_KEY = "your key"
```

That lasts for the current terminal only. To keep it:

```powershell
[Environment]::SetEnvironmentVariable("CH_API_KEY", "your key", "User")
```

Then reopen PowerShell.

## Using it

**Check the OCR engine first.**

```powershell
python cli.py engine
```

Prints which engine is configured, whether it actually answers, and at what
resolution. On the vision model this is the difference between finding out
now and finding out an hour into a batch.

**Start with a file you already have.** No API key, no rate limit, so this
is the right way to tune anything.

```powershell
python cli.py file samples\SC284050_aa_2026-01-27.pdf
```

```
10 pages, read [0, 1, 2, 3, 4, 5, 6, 7], 8 OCR calls, 24.0s

  turnover         not filed   profit and loss withheld under s444
  debtors            493,712   page 3
             from: Debtors 6 493,712 517,202
  employees               32   page 6
             from: Average number of employees, including directors, during the year was: 32 (2024 30)
```

Add `--show` to see everything it read, or `--out text.txt` to save it. Add
`--engine unlimited` to read the same file with the vision model without
changing any environment variable.

**One company, fetched live:**

```powershell
python cli.py company 09760850
```

**A batch, with a summary and a csv:**

```powershell
python cli.py batch numbers.txt --workers 4 --csv results.csv
```

The summary line that matters is the route split. Every company that comes
back as `zip-ixbrl` or `ixbrl` cost you a second and gave exact figures. Only
the `pdf` ones cost real time. Run a hundred companies representative of your
list before you size anything.

**Every tagged fact from an iXBRL filing:**

```powershell
python cli.py dump 09760850
```

## As a service

```powershell
$env:API_TOKEN = "pick something long"
python app.py
```

```powershell
curl.exe http://localhost:8080/health
```

### POST /v1/company

```json
{ "company_number": "09760850" }
```

Returns turnover, debtors, employee_size, net_assets, total_assets,
total_liabilities, the route taken, which pages were read, and the evidence
line for each figure. A flat `summary` object carries just the fields the
research sheet needs. Results are cached per company for `CACHE_DAYS` (default
30); send `"refresh": true` to read the filing again.

Total assets and total liabilities are rarely printed as single lines on a UK
balance sheet, so they are worked out and the working is returned:
total assets = total assets less current liabilities + creditors due within
one year (or fixed + current assets in iXBRL), and total liabilities = total
assets - net assets.

### The balance check

OCR mistakes are silent, so every PDF result carries `balance_check`:
`pass`, `fail` or `not checked`. It re-adds the balance sheet that net
assets was read from: the items in each section against their subtotal,
current assets less creditors against net current assets, fixed assets plus
net current assets against total assets less current liabilities, that
less the longer term liabilities against net assets, and capital and
reserves against net assets. `balance_check_detail` gives the working for
anything that failed. A `fail` does not say which figure is wrong, only
that a line on that page was misread, so look at the row before using it.
Sort a batch csv on this column first.

### POST /v1/ocr

Drop in replacement for the Mistral OCR node. Takes
`document.document_url` or `document.document_base64`, returns
`{ pages: [{ index, markdown, source }] }`, so nothing downstream changes.

### POST /v1/ixbrl

Takes iXBRL or a zip report package as base64 and returns the tagged figures.

All `/v1/` endpoints require the `X-Auth-Token` header when `API_TOKEN` is
set. Set it. If this ever sits behind a public tunnel, anyone who guesses
the address can otherwise burn your Companies House rate limit.

## Docker

```powershell
"CH_API_KEY=your key`nAPI_TOKEN=pick something long" | Out-File -Encoding utf8 .env
docker compose up -d --build
```

Compose refuses to start if either variable is missing, which is deliberate.

## n8n

Put `n8n_route_node.js` in a Code node after the document metadata request,
then an IF node on `is_ixbrl`.

- **iXBRL branch**: fetch `{{ $json.document_url }}` with header
  `Accept: {{ $json.accept }}`, POST the body to `/v1/ixbrl`, wire straight
  into Google Sheets. No OCR and no LLM.
- **PDF branch**: POST to `/v1/company` with the company number, or to
  `/v1/ocr` with the base64 document if you want to keep your existing
  downstream nodes.

Note the zip check in that node. Modern filings ship the tagged report inside
a zip package, and the old node only looked for `application/xhtml+xml`, so
every recent filing fell through to the expensive PDF branch.

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `CH_API_KEY` | none | required for anything that fetches |
| `API_TOKEN` | none | required header on `/v1/`, endpoints are open without it |
| `READ_DPI` | 200 | resolution for the pages that matter |
| `SCOUT_DPI` | 150 | resolution for the cheap first look on large filings |
| `OCR_PSM` | 6 | 6 for single column filings, 3 for multi column annual reports |
| `OCR_WORKERS` | 4 | Tesseract threads, match your core count |
| `MAX_OCR_PAGES` | 45 | ceiling per document |
| `SMALL_DOC_PAGES` | 24 | below this the scout pass is skipped |
| `TESSERACT_CMD` | auto | path to tesseract.exe on Windows |
| `OCR_ENGINE` | `tesseract` | `tesseract` or `unlimited` |
| `SCOUT_ENGINE` | same as `OCR_ENGINE` | engine for the cheap page hunt |
| `UNLIMITED_URL` | `http://127.0.0.1:10000` | where the model is served |
| `UNLIMITED_MODE` | `server` | `server` or `local` |
| `UNLIMITED_SERVER` | `sglang` | `sglang` or `vllm`, changes the payload |
| `UNLIMITED_IMAGE_MODE` | `gundam` | `gundam` tiles the page, `base` does not |
| `UNLIMITED_WORKERS` | 4 | pages in flight against the model server |
| `UNLIMITED_MODEL_DIR` | `baidu/Unlimited-OCR` | local path or HF id, local mode |
| `UNLIMITED_TIMEOUT` | 600 | seconds per page |
| `FIGURE_REPAIR` | follows the engine | `full` for Tesseract, `light` for the model |
| `MAX_JOBS` | 1 | filings read at the same time, others queue |
| `CACHE_DAYS` | 30 | reuse a company's figures this long, 0 disables |
| `CACHE_DIR` | `.` (`/data` in Docker) | where the cache file lives |

`READ_DPI` defaults to 200 on Tesseract and 300 on the vision model, which
is the resolution the model card converts PDFs at.

On Tesseract, `READ_DPI` is 200 because Companies House re renders filings as
bitonal TIFF at 200 dpi. Rendering above the source resolution costs time and
adds nothing. I measured 150, 200, 300 and 400 on the sample filing and all
four returned identical figures, at 2.0, 2.7, 3.7 and 4.6 seconds a page.
If your batch validates at 150, use it.

`OCR_PSM` is worth knowing about. 6 tells Tesseract the page is one uniform
block, which suits a small company filing. A plc annual report is a multi
column magazine layout, and psm 6 merges the columns into nonsense like
"Although full year revenues were below As reported in ast year's Annual
Report". Set `OCR_PSM=3` for those.

## Sizing

The number that decides everything is how many of your companies have iXBRL.
Those are effectively free. Run a representative sample through
`cli.py batch` and read the route split before planning for volume.

On the Companies House side the limit is 600 requests per five minute
window. This flow makes two calls per company before the document download,
and the download is served from S3 and does not count. So 5,000 companies is
roughly 10,000 calls, or about an hour and a half of steady running. The
client reads `X-Ratelimit-Remaining` and waits on it rather than using fixed
delays.

## Tests

```powershell
python test_extract.py
```

69 checks covering the separator repair, note reference exclusion, the year
column, the section 444 wording, headcounts with thousands separators, and
the sample filing end to end, plus the markdown to row conversion the
vision engine depends on. Every one of them is a mistake the previous
version made on a real document. Run this before and after any change to
the extraction rules. It needs no GPU and no model server: the markdown
tests use the output shape the model returns, so they pin the conversion
without waiting on hardware.
