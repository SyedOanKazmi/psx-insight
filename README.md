---
title: Invock Investments
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Invock Investments

A stock analysis and prediction platform for the Pakistan Stock Exchange.

- **Backend:** FastAPI (JWT auth, SQLite, scikit-learn Random Forest)
- **Frontend:** Vue 3 (Vite, Pinia, Chart.js)
- **Hosting:** single Docker image — FastAPI serves the API *and* the built Vue site.

## Run locally

```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn main:app --port 8001

# Frontend (second terminal)
cd web
npm install
npm run dev        # opens http://localhost:5173
```

## Demo accounts

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@psx.com | admin123 |
| Expert | expert@psx.com | expert123 |
| Investor | user@psx.com | user123 |
