"""
Invock Investments — FastAPI backend.

A JWT-secured REST API serving the Vue frontend. Reuses SQLite for storage and
a Random Forest per stock for predictions. Scoped to 3 stocks for now.

Run:  uvicorn main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import os
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import db
import ml
from auth import (hash_password, verify_password, create_access_token,
                  get_current_user, require_roles)
from schemas import (RegisterRequest, LoginRequest, QuestionCreate, AnswerCreate,
                     FeedbackCreate, FeedbackResponse, AnnounceCreate, RoleUpdate)

VALID_ROLES = ("investor", "expert", "admin")

app = FastAPI(title="Invock Investments API")

# Allow the Vue dev server locally, plus any deployed frontend origins listed
# in the CORS_ORIGINS env var. The regex also permits Vercel preview/prod URLs.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables, seed demo data and import prices on startup.
db.init_db()


# ─── Auth ─────────────────────────────────────────────────────────────────────
@app.post("/api/auth/register")
def register(req: RegisterRequest):
    conn = db.get_db()
    if conn.execute("SELECT 1 FROM users WHERE email=?", (req.email,)).fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")
    conn.execute(
        "INSERT INTO users (email, password, role, name, created_at) VALUES (?,?,?,?,?)",
        (req.email, hash_password(req.password), "investor", req.name,
         datetime.now().strftime("%Y-%m-%d")),
    )
    conn.commit()
    conn.close()
    db.add_notification(req.email, "Welcome to Invock Investments!", "info")
    token = create_access_token(req.email, "investor", req.name)
    return {"access_token": token, "token_type": "bearer",
            "user": {"email": req.email, "role": "investor", "name": req.name}}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    conn = db.get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user["email"], user["role"], user["name"])
    return {"access_token": token, "token_type": "bearer",
            "user": {"email": user["email"], "role": user["role"], "name": user["name"]}}


@app.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user


# ─── Stocks ───────────────────────────────────────────────────────────────────
@app.get("/api/stocks")
def stocks(user: dict = Depends(get_current_user)):
    conn = db.get_db()
    rows = conn.execute("SELECT symbol FROM watchlist WHERE user_email=?", (user["email"],)).fetchall()
    conn.close()
    tracked = {r["symbol"] for r in rows}
    out = []
    for s in db.STOCKS:
        price, change = ml.latest_quote(s["symbol"])
        out.append({**s, "price": price, "change": change, "tracked": s["symbol"] in tracked})
    return {"stocks": out}


@app.get("/api/stocks/{ticker}/history")
def stock_history(ticker: str, user: dict = Depends(get_current_user)):
    _check_ticker(ticker)
    return ml.history(ticker)


@app.get("/api/stocks/{ticker}/summary")
def stock_summary(ticker: str, user: dict = Depends(get_current_user)):
    _check_ticker(ticker)
    return ml.summary(ticker)


@app.get("/api/predict/{ticker}")
def predict(ticker: str, days: int = 30, user: dict = Depends(get_current_user)):
    _check_ticker(ticker)
    days = min(max(days, 7), 90)
    return ml.prediction_payload(ticker, days)


def _check_ticker(ticker):
    if ticker.upper() not in db.SYMBOLS:
        raise HTTPException(status_code=404, detail="Unknown ticker")


# ─── Watchlist ────────────────────────────────────────────────────────────────
@app.post("/api/watchlist/{ticker}")
def add_watch(ticker: str, user: dict = Depends(get_current_user)):
    _check_ticker(ticker)
    conn = db.get_db()
    conn.execute("INSERT OR IGNORE INTO watchlist (user_email, symbol) VALUES (?,?)",
                 (user["email"], ticker.upper()))
    conn.commit()
    conn.close()
    return {"message": "added"}


@app.delete("/api/watchlist/{ticker}")
def remove_watch(ticker: str, user: dict = Depends(get_current_user)):
    conn = db.get_db()
    conn.execute("DELETE FROM watchlist WHERE user_email=? AND symbol=?",
                 (user["email"], ticker.upper()))
    conn.commit()
    conn.close()
    return {"message": "removed"}


# ─── Notifications ────────────────────────────────────────────────────────────
@app.get("/api/notifications")
def notifications(user: dict = Depends(get_current_user)):
    """Return this user's notifications that they have not yet read/dismissed."""
    conn = db.get_db()
    rows = conn.execute(
        """SELECT * FROM notifications
           WHERE (user_email=? OR user_email IS NULL)
             AND id NOT IN (SELECT notification_id FROM notification_reads WHERE user_email=?)
           ORDER BY id DESC LIMIT 50""",
        (user["email"], user["email"]),
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return {"notifications": items, "unread": len(items)}


@app.post("/api/notifications/read")
def mark_read(user: dict = Depends(get_current_user)):
    """Mark every currently-visible notification as read for this user."""
    conn = db.get_db()
    conn.execute(
        """INSERT OR IGNORE INTO notification_reads (user_email, notification_id)
           SELECT ?, id FROM notifications WHERE user_email=? OR user_email IS NULL""",
        (user["email"], user["email"]),
    )
    conn.commit()
    conn.close()
    return {"message": "ok"}


@app.post("/api/notifications/announce")
def announce(req: AnnounceCreate, user: dict = Depends(require_roles("admin"))):
    db.add_notification(None, req.message, "announcement")
    return {"message": "sent"}


# ─── Q&A ──────────────────────────────────────────────────────────────────────
@app.get("/api/qa")
def get_qa(user: dict = Depends(get_current_user)):
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM qa_posts ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/qa")
def post_question(req: QuestionCreate, user: dict = Depends(get_current_user)):
    conn = db.get_db()
    cur = conn.execute(
        "INSERT INTO qa_posts (author, asker_email, role, question, date) VALUES (?,?,?,?,?)",
        (user["name"], user["email"], user["role"], req.question,
         datetime.now().strftime("%Y-%m-%d")),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM qa_posts WHERE id=?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


@app.post("/api/qa/{qid}/answer")
def answer_question(qid: int, req: AnswerCreate, user: dict = Depends(require_roles("expert", "admin"))):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM qa_posts WHERE id=?", (qid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE qa_posts SET answer=?, answered_by=? WHERE id=?",
                 (req.answer, user["name"], qid))
    conn.commit()
    conn.close()
    if row["asker_email"]:
        db.add_notification(row["asker_email"], f"{user['name']} answered your question.", "answer")
    return {"message": "answered"}


@app.delete("/api/qa/{qid}")
def delete_question(qid: int, user: dict = Depends(require_roles("expert", "admin"))):
    conn = db.get_db()
    conn.execute("DELETE FROM qa_posts WHERE id=?", (qid,))
    conn.commit()
    conn.close()
    return {"message": "deleted"}


# ─── Feedback ─────────────────────────────────────────────────────────────────
@app.post("/api/feedback")
def submit_feedback(req: FeedbackCreate, user: dict = Depends(get_current_user)):
    conn = db.get_db()
    conn.execute(
        "INSERT INTO feedback (user_email, name, category, message, status, created_at) VALUES (?,?,?,?,?,?)",
        (user["email"], user["name"], req.category, req.message, "open",
         datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()
    return {"message": "Thank you for your feedback!"}


@app.get("/api/feedback")
def list_feedback(user: dict = Depends(get_current_user)):
    conn = db.get_db()
    if user["role"] == "admin":
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM feedback WHERE user_email=? ORDER BY id DESC",
                            (user["email"],)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/feedback/{fid}/respond")
def respond_feedback(fid: int, req: FeedbackResponse, user: dict = Depends(require_roles("admin"))):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE feedback SET admin_response=?, status='resolved' WHERE id=?", (req.response, fid))
    conn.commit()
    conn.close()
    if row["user_email"]:
        db.add_notification(row["user_email"], "An admin responded to your feedback.", "feedback")
    return {"message": "responded"}


@app.delete("/api/feedback/{fid}")
def delete_feedback(fid: int, user: dict = Depends(require_roles("admin"))):
    conn = db.get_db()
    conn.execute("DELETE FROM feedback WHERE id=?", (fid,))
    conn.commit()
    conn.close()
    return {"message": "deleted"}


# ─── Admin ────────────────────────────────────────────────────────────────────
@app.get("/api/admin/users")
def admin_users(user: dict = Depends(require_roles("admin"))):
    conn = db.get_db()
    rows = conn.execute("SELECT email, name, role FROM users").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/admin/users/{email}/role")
def admin_set_role(email: str, req: RoleUpdate, user: dict = Depends(require_roles("admin"))):
    if req.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail="Invalid role")
    if email == user["email"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    conn = db.get_db()
    if not conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    conn.execute("UPDATE users SET role=? WHERE email=?", (req.role, email))
    conn.commit()
    conn.close()
    db.add_notification(email, f"Your account role is now '{req.role}'.", "info")
    return {"message": "updated"}


@app.delete("/api/admin/users/{email}")
def admin_delete_user(email: str, user: dict = Depends(require_roles("admin"))):
    if email == user["email"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    conn = db.get_db()
    conn.execute("DELETE FROM users WHERE email=?", (email,))
    conn.commit()
    conn.close()
    return {"message": "deleted"}


# ─── Serve the built Vue frontend (production) ────────────────────────────────
# When deployed, the built site is copied to ./static and served from the same
# server as the API (one URL). Mounted LAST so /api/* and /docs win first.
# The app uses hash routing, so serving index.html at "/" is enough.
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
