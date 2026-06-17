# ── Stage 1: build the Vue frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
# Empty API URL => the app calls the same server it is served from (relative URLs).
ENV VITE_API_URL=""
RUN npm run build

# ── Stage 2: Python backend that also serves the built frontend ───────────────
FROM python:3.12-slim
WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY data/ ./data/
# Drop the compiled Vue site where FastAPI serves it from (./api/static).
COPY --from=frontend /web/dist ./api/static

WORKDIR /app/api
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
