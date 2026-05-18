# Project knowledge

This file gives Freebuff context about your project: goals, commands, conventions, and gotchas.

## What This Project Is

**NVIDIA NIM to OpenAI API Proxy** — a Flask proxy that translates OpenAI-compatible API requests into NVIDIA NIM (NVIDIA Inference Microservices) API calls. Lets tools expecting an OpenAI endpoint transparently use NVIDIA-hosted models.

## Quickstart
- **Install:** `pip install -r requirements.txt`
- **Run (dev):** `python main.py`
- **Run (production):** `gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **Set env vars:** `NVIDIA_API_KEY`, optionally `NVIDIA_BASE_URL` (default: `https://integrate.api.nvidia.com/v1`) and `PORT` (default: `5000`)

## Architecture
- **Key files:** `main.py` — entire application in a single file (routes, proxy logic, model mappings)
- **Endpoints:**
  - `POST /v1/chat/completions` — chat completions (streaming + non-streaming)
  - `POST /v1/images/generations` — image generation
  - `GET /v1/models` — list available models
  - `GET /health` — health check
  - `GET /` — API info
- **Data flow:** OpenAI-format request → model name mapped to NVIDIA NIM model → request forwarded to NVIDIA API → response remapped back to OpenAI format

## Conventions
- **No tests:** project has no test suite
- **Single file:** all logic in `main.py`
- **Model mapping:** `MODEL_MAPPING` dict maps OpenAI model names (e.g., `gpt-4`, `deepseek-chat`) to NVIDIA NIM model IDs. **Unknown model names pass through as-is** — permite usar cualquier modelo NVIDIA NIM directamente sin configuración
- **Image mapping:** `IMAGE_MODEL_MAPPING` maps OpenAI image models (e.g., `dall-e-3`) to NVIDIA image models. Same passthrough behavior.
- **Dynamic model listing:** `GET /v1/models` ahora consulta la API de NVIDIA para mostrar modelos reales disponibles, con caché de 5 minutos. Fallback a lista estática si NVIDIA no responde.
- **Alias info:** Los modelos alias incluyen campo `aliases_to` indicando a qué modelo NVIDIA se resuelven.
- **CORS:** Wide open (all origins, methods, headers)

## Gotchas
- **Procfile references `proxy:app`** but the app lives in `main.py`. Update Procfile if renaming the module.
- **No `.env` in repo** — uses `os.environ.get()` with fallback defaults. Create `.env` with `NVIDIA_API_KEY` manually.
- **No HTTPS/TLS** — runs plain HTTP. Use a reverse proxy (e.g., nginx) in production.
- **Debug mode off** by default in `main.py` (`debug=False`).
- **Image API path** uses the mapped model directly in the URL path, not a fixed endpoint.
- **All routes are synchronous** — uses blocking `requests.post()` calls.
