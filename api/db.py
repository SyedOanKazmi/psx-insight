"""
SQLite layer for the Invock Investments FastAPI backend.

Holds users, Q&A, feedback, notifications, watchlists and price history.
Scoped to THREE stocks for now so training stays fast.
"""
import sqlite3
import os
import hashlib
from datetime import datetime

BASE_DIR   = os.path.dirname(__file__)
DB_PATH    = os.path.join(BASE_DIR, "psx.db")
PRICES_CSV = os.path.join(BASE_DIR, "../data/psx_stocks.csv")

# Only these three stocks are loaded for now (faster to train).
STOCKS = [
    {"symbol": "OGDC", "name": "Oil & Gas Development Co.", "sector": "Energy"},
    {"symbol": "HBL",  "name": "Habib Bank",               "sector": "Banking"},
    {"symbol": "LUCK", "name": "Lucky Cement",             "sector": "Cement"},
]
SYMBOLS = [s["symbol"] for s in STOCKS]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_pw(pw):
    # Demo accounts are seeded with a simple hash; real registrations use bcrypt.
    return hashlib.sha256(pw.encode()).hexdigest()


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        email      TEXT PRIMARY KEY,
        password   TEXT NOT NULL,
        role       TEXT NOT NULL,
        name       TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS qa_posts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        author      TEXT NOT NULL,
        asker_email TEXT,
        role        TEXT,
        question    TEXT NOT NULL,
        answer      TEXT,
        answered_by TEXT,
        date        TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email     TEXT,
        name           TEXT,
        category       TEXT,
        message        TEXT NOT NULL,
        status         TEXT DEFAULT 'open',
        admin_response TEXT,
        created_at     TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS notifications (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        message    TEXT NOT NULL,
        type       TEXT DEFAULT 'info',
        is_read    INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS watchlist (
        user_email TEXT NOT NULL,
        symbol     TEXT NOT NULL,
        PRIMARY KEY (user_email, symbol)
    );
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        date   TEXT NOT NULL,
        open   REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (ticker, date)
    );
    -- Per-user read state, so a notification (incl. shared broadcasts)
    -- disappears for the user who read it without affecting others.
    CREATE TABLE IF NOT EXISTS notification_reads (
        user_email      TEXT NOT NULL,
        notification_id INTEGER NOT NULL,
        PRIMARY KEY (user_email, notification_id)
    );
    ''')
    conn.commit()

    now = datetime.now().strftime("%Y-%m-%d")
    if not c.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO users (email, password, role, name, created_at) VALUES (?,?,?,?,?)",
            [
                ("admin@psx.com",  _seed_pw("admin123"),  "admin",    "Admin",           now),
                ("expert@psx.com", _seed_pw("expert123"), "expert",   "Dr. Ayesha Khan", now),
                ("user@psx.com",   _seed_pw("user123"),   "investor", "Ali Raza",        now),
            ],
        )
    if not c.execute("SELECT 1 FROM notifications LIMIT 1").fetchone():
        c.execute(
            "INSERT INTO notifications (user_email, message, type, created_at) VALUES (?,?,?,?)",
            (None, "Welcome to Invock Investments!", "announcement", now),
        )
    conn.commit()
    conn.close()
    import_prices()


def import_prices():
    """Load the three stocks' price history from the CSV (once)."""
    conn = get_db()
    c = conn.cursor()
    if c.execute("SELECT 1 FROM prices LIMIT 1").fetchone() or not os.path.exists(PRICES_CSV):
        conn.close()
        return
    import csv
    with open(PRICES_CSV, newline="") as fh:
        batch = []
        for row in csv.DictReader(fh):
            if row["Ticker"] not in SYMBOLS:
                continue
            batch.append((row["Ticker"], row["Date"][:10], float(row["Open"]),
                          float(row["High"]), float(row["Low"]), float(row["Close"]),
                          float(row["Volume"])))
            if len(batch) >= 5000:
                c.executemany("INSERT OR IGNORE INTO prices VALUES (?,?,?,?,?,?,?)", batch)
                batch = []
        if batch:
            c.executemany("INSERT OR IGNORE INTO prices VALUES (?,?,?,?,?,?,?)", batch)
    conn.commit()
    conn.close()


def add_notification(user_email, message, ntype="info"):
    conn = get_db()
    conn.execute(
        "INSERT INTO notifications (user_email, message, type, created_at) VALUES (?,?,?,?)",
        (user_email, message, ntype, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()
