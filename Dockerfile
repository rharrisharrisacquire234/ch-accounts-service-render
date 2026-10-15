FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./

# Tesseract stays in the image even when the vision model is the read
# engine, because SCOUT_ENGINE=tesseract is the cheap way to find the four
# pages worth reading in a two hundred page filing.
#
# The image deliberately has no torch in it. Unlimited-OCR runs behind vLLM
# or SGLang on a GPU box and this service talks to it over http, so the
# service stays small and can be restarted without reloading model weights.
ENV OCR_ENGINE=tesseract \
    UNLIMITED_URL=http://127.0.0.1:10000 \
    UNLIMITED_MODE=server \
    UNLIMITED_SERVER=sglang \
    UNLIMITED_IMAGE_MODE=gundam \
    UNLIMITED_WORKERS=4 \
    READ_DPI=200 \
    SCOUT_DPI=150 \
    OCR_PSM=6 \
    OCR_WORKERS=4 \
    HOST=0.0.0.0 \
    PORT=8080 \
    MAX_JOBS=1 \
    CACHE_DAYS=30 \
    CACHE_DIR=/data

# The financials cache lives here. Mount a volume so it survives rebuilds.
VOLUME /data

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import httpx,os,sys; sys.exit(0 if httpx.get('http://127.0.0.1:%s/health' % os.getenv('PORT','8080')).json()['ok'] else 1)"

# Hosts like Render choose the port through $PORT, so it is read at start-up.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
