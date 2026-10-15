# Deploying on the Hostinger VPS

The service runs on the CPU with Tesseract, which is free. No GPU is needed and
none is available on a Hostinger VPS, so the `unlimited_ocr` (GPU) service in
`compose.yaml` is never started. Most filings since about 2023 are iXBRL and are
read without any OCR in a second or two. OCR only runs for scanned PDFs.

## 1. Check the VPS

SSH in (hPanel > VPS > Overview shows the IP and root login):

```bash
ssh root@YOUR_VPS_IP
nproc            # vCPUs  -> OCR_CPUS and OCR_WORKERS
free -h          # RAM    -> 2G minimum free for the service, 3G is comfortable
docker --version || curl -fsSL https://get.docker.com | sh
docker compose version
```

## 2. Upload the code

From your own machine (PowerShell):

```powershell
scp ch-accounts-service.zip root@YOUR_VPS_IP:/opt/
```

On the VPS:

```bash
apt-get install -y unzip
cd /opt && unzip ch-accounts-service.zip && cd ch-accounts-service
```

## 3. Settings

```bash
cp .env.example .env
openssl rand -hex 32          # copy this into API_TOKEN
nano .env
```

| Setting | Value |
| --- | --- |
| `CH_API_KEY` | A Companies House key. Ideally a **second** key, so this service and the n8n workflow do not share one 600 requests / 5 minutes limit. |
| `API_TOKEN` | The random string from `openssl`. The same value goes into n8n's Config node as `OCR_API_TOKEN`. |
| `OCR_CPUS`, `OCR_WORKERS` | The `nproc` number (e.g. 2). `OCR_CPUS` above the real core count stops the container starting. |
| `OCR_MEMORY` | `3G` on an 8 GB plan, `2G` on a 4 GB plan. |
| `MAX_JOBS` | `1` on 2 to 4 vCPUs. Other requests wait in a queue rather than slowing each other down. |

## 4. Start it

Pick the case that matches where n8n runs.

**A. n8n runs in Docker on this same VPS (most likely).**
Put the service on n8n's Docker network, so n8n reaches it by name and no port is
opened to the internet.

```bash
docker ps --format '{{.Names}}'                                 # find the n8n container
docker inspect <n8n-container> --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}'
echo "N8N_NETWORK=<network name printed above>" >> .env
docker compose -f compose.yaml -f compose.n8n.yaml up -d --build
```

n8n Config node: `OCR_SERVICE_URL: 'http://ch_accounts:8080'`

**B. n8n is installed directly on this VPS (no Docker).**

```bash
docker compose up -d --build
```

n8n Config node: `OCR_SERVICE_URL: 'http://127.0.0.1:8080'`

**C. n8n runs somewhere else (n8n Cloud, another server).**
The service must be reachable over HTTPS. Point a subdomain (e.g. `ocr.yourdomain.com`)
at the VPS IP, then, if nothing else is using ports 80/443:

```bash
docker compose up -d --build
apt-get install -y caddy
cat > /etc/caddy/Caddyfile <<'CADDY'
ocr.yourdomain.com {
    reverse_proxy 127.0.0.1:8080
}
CADDY
systemctl reload caddy
```

n8n Config node: `OCR_SERVICE_URL: 'https://ocr.yourdomain.com'`.
If ports 80/443 are already used (for example by the proxy in front of n8n), add a
route for the subdomain to that proxy instead of installing Caddy.

## 5. Check it works

```bash
docker compose ps                       # STATUS should become "healthy"
curl -s localhost:8080/health
TOKEN=$(grep ^API_TOKEN .env | cut -d= -f2)
curl -s -X POST localhost:8080/v1/company \
  -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"company_number":"01411253"}' | python3 -m json.tool | sed -n '/"summary"/,/}/p'
docker compose exec ocr python test_extract.py      # rule tests, no network needed
```

For case A, also check n8n can see it:

```bash
docker exec <n8n-container> wget -qO- http://ch_accounts:8080/health
```

(`health` needs no token. Everything under `/v1/` does.)

## 6. Day to day

| Task | Command |
| --- | --- |
| Logs | `docker compose logs -f ocr` |
| Restart | `docker compose restart ocr` |
| Deploy new code | upload, unzip over the folder, then the same `up -d --build` command as step 4 |
| Re-read one company | send `"refresh": true` in the request body |
| Clear the whole cache | `docker compose exec ocr rm -f /data/financials_cache.sqlite` |

## Timing on a 2 vCPU VPS

| Filing | Typical time |
| --- | --- |
| iXBRL (most filings from 2023 on) | 1 to 3 s |
| Text PDF | a few seconds |
| Scanned small company PDF, about 10 pages | 20 to 40 s |
| Scanned large group PDF | several minutes (capped by `MAX_OCR_PAGES`) |
| Any company already read in the last `CACHE_DAYS` | instant |

The n8n node waits up to `FINANCIALS_TIMEOUT_SECONDS` (600) per company. If your
n8n has an overall execution timeout (`EXECUTIONS_TIMEOUT`), make sure it allows
for this.
