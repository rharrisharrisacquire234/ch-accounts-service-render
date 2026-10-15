"""
Talking to baidu/Unlimited-OCR.

https://github.com/baidu/Unlimited-OCR

Two ways in, and the choice matters more than it looks.

server   The model runs behind vLLM or SGLang and speaks the OpenAI chat
         completions API. This service then needs nothing but httpx, keeps
         its 300MB image, and the GPU box can sit anywhere. Set
         UNLIMITED_URL and you are done. This is the route to use.

local    transformers loads the weights into this process. Simple to start,
         but it drags torch, a CUDA build and roughly 8GB of GPU memory into
         the same container as the web service, and it serialises requests.
         Only worth it for a single machine test.

Either way the model needs an NVIDIA GPU. There is no CPU path that
finishes in a useful time, so if the machine has no GPU, keep
OCR_ENGINE=tesseract and point UNLIMITED_URL at a box that has one.

The prompts and sampling settings below come from the model card. They are
not decoration: no_repeat_ngram_size is what stops the model looping on a
page of repeated table rows, which a set of accounts is full of.
"""

import base64
import io
import os
import threading
from typing import List, Optional

import httpx
from PIL import Image

UNLIMITED_URL = os.getenv("UNLIMITED_URL", "http://127.0.0.1:10000")
UNLIMITED_MODEL = os.getenv("UNLIMITED_MODEL", "Unlimited-OCR")
UNLIMITED_MODEL_DIR = os.getenv("UNLIMITED_MODEL_DIR", "baidu/Unlimited-OCR")
UNLIMITED_MODE = os.getenv("UNLIMITED_MODE", "server")        # server | local
UNLIMITED_SERVER = os.getenv("UNLIMITED_SERVER", "sglang")    # sglang | vllm
UNLIMITED_IMAGE_MODE = os.getenv("UNLIMITED_IMAGE_MODE", "gundam")
UNLIMITED_PROMPT = os.getenv("UNLIMITED_PROMPT", "document parsing.")
UNLIMITED_TIMEOUT = float(os.getenv("UNLIMITED_TIMEOUT", "600"))
UNLIMITED_MAX_TOKENS = int(os.getenv("UNLIMITED_MAX_TOKENS", "32768"))
UNLIMITED_NGRAM = int(os.getenv("UNLIMITED_NGRAM", "35"))
UNLIMITED_NGRAM_WINDOW = int(os.getenv("UNLIMITED_NGRAM_WINDOW", "128"))
UNLIMITED_RETRIES = int(os.getenv("UNLIMITED_RETRIES", "2"))
UNLIMITED_DEVICE = os.getenv("UNLIMITED_DEVICE", "auto")   # auto | cuda | cpu
UNLIMITED_DTYPE = os.getenv("UNLIMITED_DTYPE", "auto")
UNLIMITED_4BIT = os.getenv("UNLIMITED_4BIT", "0") not in ("0", "", "false")

# Weights are about 6.7GB at bfloat16. Add the vision encoder activations
# and a KV cache for a long page and you want roughly 10GB of free VRAM.
# Below that the load either fails outright or spills to the CPU and runs at
# CPU speed while still occupying the card.
VRAM_WANTED_GB = float(os.getenv("UNLIMITED_VRAM_WANTED_GB", "10"))


def encode_png(img: Image.Image) -> str:
    buffer = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class UnlimitedServer:
    """The model behind vLLM or SGLang, one page per request."""

    def __init__(self, url: str = UNLIMITED_URL, model: str = UNLIMITED_MODEL,
                 flavour: str = UNLIMITED_SERVER):
        self.url = url.rstrip("/")
        self.model = model
        self.flavour = flavour.lower()
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()
        # Set once the server has rejected the SGLang only fields, so we do
        # not pay for the same round trip on every page after the first.
        self._plain = self.flavour != "sglang"

    def client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(UNLIMITED_TIMEOUT, connect=15.0),
                    limits=httpx.Limits(max_connections=32,
                                        max_keepalive_connections=16),
                    trust_env=False)
            return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def check(self) -> str:
        """Fail once, at the top, with something you can act on."""
        try:
            response = self.client().get(f"{self.url}/v1/models", timeout=15.0)
            response.raise_for_status()
            names = [m.get("id") for m in response.json().get("data", [])]
        except Exception as exc:
            raise RuntimeError(
                f"Unlimited-OCR server not answering at {self.url}: {exc}\n"
                "Start it on the GPU box first, for example:\n"
                "  docker run --gpus all -p 10000:8000 "
                "vllm/vllm-openai:unlimited-ocr --model baidu/Unlimited-OCR\n"
                "then set UNLIMITED_URL to that address.")
        if names and self.model not in names:
            # Not fatal. Servers are often started with a different served
            # name, and using the one it reports is friendlier than refusing.
            self.model = names[0]
        return f"unlimited-ocr @ {self.url} ({self.model})"

    def _payload(self, image_b64: str, plain: bool) -> dict:
        content = [
            {"type": "text", "text": UNLIMITED_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": UNLIMITED_MAX_TOKENS,
            "stream": False,
        }
        if plain:
            return payload

        # SGLang specific. The repeat blocker is a custom logit processor
        # there, and the image mode chooses gundam tiling over base.
        from_sglang = {
            "skip_special_tokens": False,
            "images_config": {"image_mode": UNLIMITED_IMAGE_MODE},
            "custom_params": {"ngram_size": UNLIMITED_NGRAM,
                              "window_size": UNLIMITED_NGRAM_WINDOW},
        }
        try:
            from sglang.srt.sampling.custom_logit_processor import (
                DeepseekOCRNoRepeatNGramLogitProcessor)
            from_sglang["custom_logit_processor"] = \
                DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        except Exception:
            # The sglang package is only needed to serialise that class. If
            # it is not installed here, the server still runs, we just lose
            # the repeat guard rather than the whole request.
            pass
        payload.update(from_sglang)
        return payload

    def read(self, img: Image.Image) -> str:
        image_b64 = encode_png(img)
        last = None
        for attempt in range(UNLIMITED_RETRIES + 1):
            plain = self._plain
            try:
                response = self.client().post(
                    f"{self.url}/v1/chat/completions",
                    json=self._payload(image_b64, plain))
                if response.status_code == 400 and not plain:
                    # vLLM and plain OpenAI servers reject the extra fields.
                    self._plain = True
                    continue
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                last = exc
                if attempt >= UNLIMITED_RETRIES:
                    break
        raise RuntimeError(f"unlimited-ocr request failed: {last}")


def torch_report() -> dict:
    """What torch actually has available, which is not always what was
    intended. `pip install torch` on Windows gives the CPU only build, and
    the failure it produces later says nothing about the missing index url,
    so it is worth stating plainly up front."""
    out = {"torch": None, "cuda_build": None, "cuda_available": False,
           "devices": [], "capability": None, "vram_total_gb": None,
           "vram_free_gb": None, "bf16": None, "note": None}
    try:
        import torch
    except Exception as exc:
        out["note"] = f"torch not importable: {exc}"
        return out
    out["torch"] = torch.__version__
    out["cuda_build"] = getattr(torch.version, "cuda", None)
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["devices"] = [torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())]
            major, minor = torch.cuda.get_device_capability(0)
            out["capability"] = f"{major}.{minor}"
            # bfloat16 needs Ampere. A Turing card reports 7.x and has to
            # run float16 instead, which the model card does not mention
            # because it assumes a datacentre card.
            out["bf16"] = major >= 8
            free, total = torch.cuda.mem_get_info(0)
            out["vram_free_gb"] = round(free / 1024 ** 3, 1)
            out["vram_total_gb"] = round(total / 1024 ** 3, 1)
    except Exception as exc:
        out["note"] = str(exc)
    if not out["cuda_build"]:
        out["note"] = ("this is the CPU only build of torch, so no GPU can "
                       "be used no matter what card is in the machine")
    elif not out["cuda_available"]:
        out["note"] = ("torch has CUDA support but no usable device, so "
                       "check the driver with nvidia-smi")
    elif out["vram_free_gb"] is not None and \
            out["vram_free_gb"] < VRAM_WANTED_GB:
        out["note"] = (
            f"{out['vram_free_gb']}GB free of {out['vram_total_gb']}GB. "
            f"The weights alone are 6.7GB and about {VRAM_WANTED_GB:g}GB "
            "free is wanted. This card is too small to hold the model.")
    elif not out["bf16"]:
        out["note"] = ("compute capability " + str(out["capability"]) +
                       ", so bfloat16 is not supported. float16 will be "
                       "used instead.")
    return out


def pick_device(requested: str = UNLIMITED_DEVICE) -> str:
    requested = (requested or "auto").lower().strip()
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class UnlimitedLocal:
    """transformers in this process. One device, one page at a time."""

    def __init__(self, model_dir: str = UNLIMITED_MODEL_DIR):
        self.model_dir = model_dir
        self.device = pick_device()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True)
        # bfloat16 on a GPU, float32 on a CPU. Half precision on a CPU is
        # not a saving, it is emulated and slower than the full width path.
        dtype = self._dtype(torch)
        kwargs = dict(trust_remote_code=True, use_safetensors=True,
                      dtype=dtype)
        if UNLIMITED_4BIT:
            # Last resort for a small card. Quantising the vision encoder
            # costs accuracy on exactly the thing we are testing, so treat
            # any result from this as indicative and nothing more.
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
        try:
            model = AutoModel.from_pretrained(self.model_dir, **kwargs)
        except TypeError:
            # older transformers wants the deprecated spelling
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModel.from_pretrained(self.model_dir, **kwargs)
        model = model.eval()
        # A quantised model is already placed by device_map and moving it
        # again throws.
        self._model = model if UNLIMITED_4BIT else model.to(self.device)

    def _dtype_name(self) -> str:
        wanted = UNLIMITED_DTYPE.lower().strip()
        if wanted != "auto":
            return wanted
        return "float32" if self.device != "cuda" else "float16"

    def _dtype(self, torch):
        wanted = UNLIMITED_DTYPE.lower().strip()
        if wanted in ("bfloat16", "bf16"):
            return torch.bfloat16
        if wanted in ("float16", "fp16", "half"):
            return torch.float16
        if wanted in ("float32", "fp32"):
            return torch.float32
        if self.device != "cuda":
            return torch.float32
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16

    def check(self) -> str:
        report = torch_report()
        if self.device == "cuda" and not report["cuda_available"]:
            raise RuntimeError(
                f"UNLIMITED_DEVICE=cuda but no CUDA device is usable.\n"
                f"  torch {report['torch']}, cuda build "
                f"{report['cuda_build'] or 'none'}\n"
                f"  {report['note']}")

        # Fail here rather than after a six minute load. Without this check
        # the card runs out of memory somewhere inside the model's own code
        # and reports whatever it happened to be doing at the time, which
        # sends you chasing dependencies instead of the actual problem.
        free = report["vram_free_gb"]
        if self.device == "cuda" and free is not None and not UNLIMITED_4BIT:
            needed = 7.5      # weights at float16 plus working room
            if free < needed:
                raise RuntimeError(
                    f"this GPU cannot hold the model.\n"
                    f"  {free}GB free of {report['vram_total_gb']}GB, and "
                    f"the weights alone are 6.7GB.\n"
                    f"  Three ways forward:\n"
                    f"    UNLIMITED_DEVICE=cpu     works, minutes per page, "
                    f"enough to judge one page\n"
                    f"    UNLIMITED_4BIT=1         squeezes it to about "
                    f"2GB, quality is indicative only\n"
                    f"    UNLIMITED_MODE=server    rent a GPU and point "
                    f"UNLIMITED_URL at it")

        if self.device == "cpu":
            wanted = 15 if self._dtype_name() == "float32" else 9
            try:
                import psutil
                spare = psutil.virtual_memory().available / 1024 ** 3
                if spare < wanted:
                    raise RuntimeError(
                        f"only {spare:.1f}GB of RAM free and about "
                        f"{wanted}GB is needed at "
                        f"{self._dtype_name()}.\n"
                        f"  Close what you can, or set "
                        f"UNLIMITED_DTYPE=bfloat16 to halve it.")
            except ImportError:
                pass
        try:
            self._load()
        except Exception as exc:
            raise RuntimeError(
                f"could not load {self.model_dir} on {self.device}: {exc}\n"
                f"  torch {report['torch']}, cuda build "
                f"{report['cuda_build'] or 'none'}, "
                f"cuda available {report['cuda_available']}\n"
                f"  {report['note'] or ''}")
        detail = f"unlimited-ocr local on {self.device} ({self.model_dir})"
        if UNLIMITED_4BIT:
            detail += "  [4 bit, quality is indicative only]"
        if self.device == "cpu":
            detail += ("  [CPU: expect minutes per page, fine for looking at "
                       "one page, not for a batch]")
        return detail

    def read(self, img: Image.Image) -> str:
        import tempfile
        self._load()
        # gundam tiles the page into crops, which multiplies the work. On a
        # CPU that turns minutes into tens of minutes, so drop to base.
        gundam = UNLIMITED_IMAGE_MODE == "gundam" and self.device != "cpu"
        with tempfile.TemporaryDirectory(prefix="ocr_page_") as work:
            page = os.path.join(work, "page.png")
            img.save(page)
            # The model is not thread safe, and a second forward pass while
            # the first is running is the fastest way to run out of VRAM.
            with self._lock:
                out = self._model.infer(
                    self._tokenizer,
                    prompt=f"<image>{UNLIMITED_PROMPT}",
                    image_file=page,
                    output_path=work,
                    base_size=1024,
                    image_size=640 if gundam else 1024,
                    crop_mode=gundam,
                    max_length=UNLIMITED_MAX_TOKENS,
                    no_repeat_ngram_size=UNLIMITED_NGRAM,
                    ngram_window=UNLIMITED_NGRAM_WINDOW,
                    save_results=True,
                )
            if isinstance(out, str) and out.strip():
                return out
            # Older builds write the result out rather than returning it.
            for name in sorted(os.listdir(work)):
                if name.endswith((".mmd", ".md", ".txt")):
                    with open(os.path.join(work, name), encoding="utf-8",
                              errors="ignore") as fh:
                        return fh.read()
        return ""


def build(mode: str = UNLIMITED_MODE):
    return UnlimitedLocal() if mode.lower() == "local" else UnlimitedServer()
