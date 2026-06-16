"""
SQLite persistence layer for PSX Insight.

Holds users, Q&A posts, feedback, notifications and per-user watchlists.
The database file lives next to this module and is created + seeded on first
run via init_db(). All other modules talk to the DB through the small helper
functions at the bottom of this file rather than writing raw SQL inline.
"""
import sqlite3
import os
import hashlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'psx.db')
PRICES_CSV = os.path.join(os.path.dirname(__file__), '../data/psx_stocks.csv')


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Curated KSE-100 constituents (used by the filtering / watchlist module) ──
# We only hold price history for the KSE-100 index itself, so individual stock
# rows are reference data: symbol, company name and sector. Users build a
# personal watchlist from this list and filter it by sector / search.
STOCKS = [
    {"symbol": "OGDC",   "name": "Oil & Gas Development Co.",   "sector": "Energy"},
    {"symbol": "PPL",    "name": "Pakistan Petroleum",          "sector": "Energy"},
    {"symbol": "PSO",    "name": "Pakistan State Oil",          "sector": "Energy"},
    {"symbol": "MARI",   "name": "Mari Petroleum",              "sector": "Energy"},
    {"symbol": "HBL",    "name": "Habib Bank",                  "sector": "Banking"},
    {"symbol": "UBL",    "name": "United Bank",                 "sector": "Banking"},
    {"symbol": "MCB",    "name": "MCB Bank",                    "sector": "Banking"},
    {"symbol": "MEBL",   "name": "Meezan Bank",                 "sector": "Banking"},
    {"symbol": "LUCK",   "name": "Lucky Cement",                "sector": "Cement"},
    {"symbol": "DGKC",   "name": "D.G. Khan Cement",            "sector": "Cement"},
    {"symbol": "ENGRO",  "name": "Engro Corporation",           "sector": "Conglomerate"},
    {"symbol": "FFC",    "name": "Fauji Fertilizer",            "sector": "Fertilizer"},
    {"symbol": "EFERT",  "name": "Engro Fertilizers",           "sector": "Fertilizer"},
    {"symbol": "HUBC",   "name": "Hub Power",                   "sector": "Power"},
    {"symbol": "PTC",    "name": "Pakistan Telecommunication",  "sector": "Telecom"},
    {"symbol": "SYS",    "name": "Systems Limited",             "sector": "Technology"},
    {"symbol": "TRG",    "name": "TRG Pakistan",                "sector": "Technology"},
    {"symbol": "NESTLE", "name": "Nestle Pakistan",             "sector": "FMCG"},
    {"symbol": "INDU",   "name": "Indus Motor Company",         "sector": "Automobile"},
    {"symbol": "SEARL",  "name": "The Searle Company",          "sector": "Pharma"},
]

SECTORS = sorted({s["sector"] for s in STOCKS})


def init_db():
    """Create tables if missing and seed demo data on a fresh database."""
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
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL,
        volume REAL,
        PRIMARY KEY (ticker, date)
    );
    ''')
    conn.commit()

    now = datetime.now().strftime('%Y-%m-%d')

    # Seed demo users (only if the table is empty)
    if not c.execute('SELECT 1 FROM users LIMIT 1').fetchone():
        c.executemany(
            'INSERT INTO users (email, password, role, name, created_at) VALUES (?,?,?,?,?)',
            [
                ("admin@psx.com",  hash_pw("admin123"),  "admin",    "Admin",          now),
                ("expert@psx.com", hash_pw("expert123"), "expert",   "Dr. Ayesha Khan", now),
                ("user@psx.com",   hash_pw("user123"),   "investor", "Ali Raza",        now),
            ],
        )

    # Seed demo Q&A
    if not c.execute('SELECT 1 FROM qa_posts LIMIT 1').fetchone():
        c.executemany(
            '''INSERT INTO qa_posts (author, asker_email, role, question, answer, answered_by, date)
               VALUES (?,?,?,?,?,?,?)''',
            [
                ("Ali Raza", "user@psx.com", "investor",
                 "Is OGDC a good long-term investment?",
                 "OGDC has strong fundamentals with consistent dividend payouts. However, watch for oil price volatility.",
                 "Dr. Ayesha Khan", "2024-12-01"),
                ("Sara Ahmed", None, "investor",
                 "What is the outlook for PSX banking sector in 2025?",
                 None, None, "2024-12-10"),
            ],
        )

    # Seed a welcome broadcast notification
    if not c.execute('SELECT 1 FROM notifications LIMIT 1').fetchone():
        c.execute(
            'INSERT INTO notifications (user_email, message, type, created_at) VALUES (?,?,?,?)',
            (None, "Welcome to PSX Insight! Explore predictions, ask experts and build your watchlist.",
             "announcement", now),
        )

    conn.commit()
    conn.close()

    import_prices()


# ─── Price history import ─────────────────────────────────────────────────────
def import_prices():
    """Load real per-stock OHLCV history from the CSV into the prices table.

    Runs once on a fresh database (skips if prices already exist). Reads in
    chunks so the import stays light on memory.
    """
    conn = get_db()
    c = conn.cursor()
    if c.execute('SELECT 1 FROM prices LIMIT 1').fetchone():
        conn.close()
        return
    if not os.path.exists(PRICES_CSV):
        conn.close()
        return

    import csv
    with open(PRICES_CSV, newline='') as fh:
        reader = csv.DictReader(fh)
        batch = []
        for row in reader:
            batch.append((row['Ticker'], row['Date'][:10],
                          float(row['Open']), float(row['High']), float(row['Low']),
                          float(row['Close']), float(row['Volume'])))
            if len(batch) >= 5000:
                c.executemany('INSERT OR IGNORE INTO prices VALUES (?,?,?,?,?,?,?)', batch)
                batch = []
        if batch:
            c.executemany('INSERT OR IGNORE INTO prices VALUES (?,?,?,?,?,?,?)', batch)
    conn.commit()
    conn.close()


# ─── Notification helper ──────────────────────────────────────────────────────
def add_notification(user_email, message, ntype="info"):
    """Insert a notification. user_email=None makes it a broadcast to everyone."""
    conn = get_db()
    conn.execute(
        'INSERT INTO notifications (user_email, message, type, created_at) VALUES (?,?,?,?)',
        (user_email, message, ntype, datetime.now().strftime('%Y-%m-%d %H:%M')),
    )
    conn.commit()
    conn.close()
