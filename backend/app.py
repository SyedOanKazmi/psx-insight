from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from functools import wraps
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import os
from datetime import datetime, timedelta
import warnings

import db
from db import get_db, hash_pw, add_notification, STOCKS, SECTORS

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
# Use a stable secret from the environment so sessions survive restarts in
# production; fall back to a random one for local development.
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32).hex())

# Create tables and seed demo data on startup.
db.init_db()

# ─── Stock data & model cache ─────────────────────────────────────────────────
FEATURED = {s['symbol'] for s in STOCKS}
DEFAULT_TICKER = 'OGDC'
model_cache = {}  # keyed by ticker


def resolve_ticker(value):
    """Return a valid featured ticker, falling back to the default."""
    value = (value or '').upper().strip()
    return value if value in FEATURED else DEFAULT_TICKER


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('role') not in roles:
                return jsonify({"error": "Unauthorized"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ─── Load & process stock data ────────────────────────────────────────────────
def load_stock_data(ticker):
    conn = get_db()
    rows = conn.execute(
        'SELECT date, open, high, low, close, volume FROM prices WHERE ticker=? ORDER BY date',
        (ticker,),
    ).fetchall()
    conn.close()
    df = pd.DataFrame([tuple(r) for r in rows],
                      columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def latest_quote(ticker):
    """Latest close and day-over-day % change for a ticker."""
    conn = get_db()
    rows = conn.execute(
        'SELECT close FROM prices WHERE ticker=? ORDER BY date DESC LIMIT 2', (ticker,),
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    price = round(rows[0]['close'], 2)
    change = round((rows[0]['close'] - rows[1]['close']) / rows[1]['close'] * 100, 2) if len(rows) > 1 else 0.0
    return price, change


def engineer_features(df):
    df = df.copy()
    df['MA7']  = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD']  = df['EMA12'] - df['EMA26']
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(14).std()
    df['Price_Range'] = df['High'] - df['Low']
    df['Lag1'] = df['Close'].shift(1)
    df['Lag5'] = df['Close'].shift(5)
    df['Lag10'] = df['Close'].shift(10)
    df.dropna(inplace=True)
    return df


def train_model(df):
    features = ['Open','High','Low','Volume','MA7','MA21','MA50','MACD','Volatility','Price_Range','Lag1','Lag5','Lag10']
    target   = 'Close'
    X = df[features]
    y = df[target]
    split = int(len(df) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return model, features, mape, X_test, y_test, y_pred


def get_model(ticker):
    if ticker not in model_cache:
        df = load_stock_data(ticker)
        df_feat = engineer_features(df)
        model, features, mape, X_test, y_test, y_pred = train_model(df_feat)
        model_cache[ticker] = {
            'model':    model,
            'features': features,
            'mape':     mape,
            'df':       df,
            'df_feat':  df_feat,
            'X_test':   X_test,
            'y_test':   y_test.values,
            'y_pred':   y_pred,
        }
    return model_cache[ticker]


def predict_future(model, df_feat, features, days=30):
    last_row = df_feat.iloc[-1].copy()
    predictions = []
    last_close  = last_row['Close']
    last_date   = df_feat['Date'].iloc[-1] if 'Date' in df_feat.columns else datetime.today()
    for i in range(1, days + 1):
        row = last_row[features].values.reshape(1, -1)
        pred = model.predict(row)[0]
        predictions.append({
            "date":  (last_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            "price": round(pred, 2)
        })
        # shift lag features
        last_row['Lag10'] = last_row['Lag5']
        last_row['Lag5']  = last_row['Lag1']
        last_row['Lag1']  = pred
        last_row['Close'] = pred
        last_row['Open']  = pred
    return predictions


# ─── Pages ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=session['name'], role=session['role'])


@app.route('/predict-page')
@login_required
def predict_page():
    return render_template('predict.html', user=session['name'], role=session['role'])


@app.route('/watchlist')
@login_required
def watchlist_page():
    return render_template('watchlist.html', user=session['name'], role=session['role'])


@app.route('/qa')
@login_required
def qa_page():
    return render_template('qa.html', user=session['name'], role=session['role'])


@app.route('/feedback')
@login_required
def feedback_page():
    return render_template('feedback.html', user=session['name'], role=session['role'])


@app.route('/admin')
@login_required
def admin_page():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    return render_template('admin.html', user=session['name'], role=session['role'])


# ─── Auth API ─────────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email', '').strip()
    pw    = data.get('password', '')
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    conn.close()
    if not user or user['password'] != hash_pw(pw):
        return jsonify({"error": "Invalid credentials"}), 401
    session['user'] = email
    session['role'] = user['role']
    session['name'] = user['name']
    return jsonify({"message": "ok", "role": user['role'], "name": user['name']})


@app.route('/api/register', methods=['POST'])
def api_register():
    data  = request.json
    email = data.get('email', '').strip()
    pw    = data.get('password', '')
    name  = data.get('name', '').strip()
    if not email or not pw or not name:
        return jsonify({"error": "All fields required"}), 400
    conn = get_db()
    exists = conn.execute('SELECT 1 FROM users WHERE email=?', (email,)).fetchone()
    if exists:
        conn.close()
        return jsonify({"error": "Email already registered"}), 409
    conn.execute(
        'INSERT INTO users (email, password, role, name, created_at) VALUES (?,?,?,?,?)',
        (email, hash_pw(pw), 'investor', name, datetime.now().strftime('%Y-%m-%d')),
    )
    conn.commit()
    conn.close()
    session['user'] = email
    session['role'] = 'investor'
    session['name'] = name
    add_notification(email, "Welcome to PSX Insight! Start by exploring the market dashboard.", "info")
    return jsonify({"message": "ok", "role": "investor", "name": name})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({"message": "ok"})


# ─── Stock Data API ───────────────────────────────────────────────────────────
@app.route('/api/stock/history')
@login_required
def stock_history():
    ticker = resolve_ticker(request.args.get('ticker'))
    df = load_stock_data(ticker)
    recent = df.tail(365)
    return jsonify({
        "ticker":  ticker,
        "dates":   recent['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "open":    recent['Open'].tolist(),
        "close":   recent['Close'].tolist(),
        "high":    recent['High'].tolist(),
        "low":     recent['Low'].tolist(),
        "volume":  recent['Volume'].tolist(),
    })


@app.route('/api/stock/summary')
@login_required
def stock_summary():
    ticker = resolve_ticker(request.args.get('ticker'))
    df = load_stock_data(ticker)
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    change = round(latest['Close'] - prev['Close'], 2)
    pct    = round((change / prev['Close']) * 100, 2)
    return jsonify({
        "ticker":       ticker,
        "latest_close": round(latest['Close'], 2),
        "change":       change,
        "pct_change":   pct,
        "high_52w":     round(df.tail(252)['High'].max(), 2),
        "low_52w":      round(df.tail(252)['Low'].min(), 2),
        "avg_volume":   int(df.tail(30)['Volume'].mean()),
        "last_date":    latest['Date'].strftime('%Y-%m-%d'),
    })


# ─── Prediction API ───────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['GET'])
@login_required
def api_predict():
    ticker = resolve_ticker(request.args.get('ticker'))
    days = int(request.args.get('days', 30))
    days = min(max(days, 7), 90)
    cache = get_model(ticker)
    future = predict_future(cache['model'], cache['df_feat'], cache['features'], days)
    accuracy = round((1 - cache['mape']) * 100, 2)

    # historical actuals vs predictions on test set
    test_dates = cache['df_feat'].iloc[-len(cache['y_test']):]['Date'].dt.strftime('%Y-%m-%d').tolist()
    return jsonify({
        "ticker":         ticker,
        "accuracy":       accuracy,
        "future":         future,
        "test_dates":     test_dates[-60:],
        "test_actual":    [round(v,2) for v in cache['y_test'][-60:]],
        "test_predicted": [round(v,2) for v in cache['y_pred'][-60:]],
    })


# ─── Stock Filtering / Watchlist API (Module 7) ───────────────────────────────
@app.route('/api/stocks')
@login_required
def api_stocks():
    """All reference stocks, annotated with whether they are on the user's watchlist."""
    conn = get_db()
    rows = conn.execute('SELECT symbol FROM watchlist WHERE user_email=?', (session['user'],)).fetchall()
    conn.close()
    tracked = {r['symbol'] for r in rows}
    out = []
    for s in STOCKS:
        price, change = latest_quote(s['symbol'])
        out.append({**s, "price": price, "change": change, "tracked": s['symbol'] in tracked})
    return jsonify({"stocks": out, "sectors": SECTORS})


@app.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    conn = get_db()
    rows = conn.execute('SELECT symbol FROM watchlist WHERE user_email=?', (session['user'],)).fetchall()
    conn.close()
    tracked = {r['symbol'] for r in rows}
    out = []
    for s in STOCKS:
        if s['symbol'] in tracked:
            price, change = latest_quote(s['symbol'])
            out.append({**s, "price": price, "change": change})
    return jsonify(out)


@app.route('/api/watchlist/<symbol>', methods=['POST'])
@login_required
def add_watchlist(symbol):
    if symbol not in {s['symbol'] for s in STOCKS}:
        return jsonify({"error": "Unknown symbol"}), 404
    conn = get_db()
    conn.execute('INSERT OR IGNORE INTO watchlist (user_email, symbol) VALUES (?,?)',
                 (session['user'], symbol))
    conn.commit()
    conn.close()
    return jsonify({"message": "added", "symbol": symbol})


@app.route('/api/watchlist/<symbol>', methods=['DELETE'])
@login_required
def remove_watchlist(symbol):
    conn = get_db()
    conn.execute('DELETE FROM watchlist WHERE user_email=? AND symbol=?', (session['user'], symbol))
    conn.commit()
    conn.close()
    return jsonify({"message": "removed", "symbol": symbol})


# ─── Notifications API (Module 8) ─────────────────────────────────────────────
@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    """Personal notifications plus broadcasts (user_email IS NULL)."""
    conn = get_db()
    rows = conn.execute(
        '''SELECT * FROM notifications
           WHERE user_email=? OR user_email IS NULL
           ORDER BY id DESC LIMIT 50''',
        (session['user'],),
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    unread = sum(1 for r in items if not r['is_read'])
    return jsonify({"notifications": items, "unread": unread})


@app.route('/api/notifications/read', methods=['POST'])
@login_required
def mark_notifications_read():
    conn = get_db()
    conn.execute(
        'UPDATE notifications SET is_read=1 WHERE user_email=?',
        (session['user'],),
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "ok"})


@app.route('/api/notifications/announce', methods=['POST'])
@login_required
def announce():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    msg = (request.json or {}).get('message', '').strip()
    if not msg:
        return jsonify({"error": "Message required"}), 400
    add_notification(None, msg, "announcement")
    return jsonify({"message": "sent"})


# ─── Q&A API ──────────────────────────────────────────────────────────────────
@app.route('/api/qa', methods=['GET'])
@login_required
def get_qa():
    conn = get_db()
    rows = conn.execute('SELECT * FROM qa_posts ORDER BY id ASC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/qa', methods=['POST'])
@login_required
def post_question():
    data = request.json
    q = data.get('question', '').strip()
    if not q:
        return jsonify({"error": "Question required"}), 400
    conn = get_db()
    cur = conn.execute(
        '''INSERT INTO qa_posts (author, asker_email, role, question, answer, answered_by, date)
           VALUES (?,?,?,?,?,?,?)''',
        (session['name'], session['user'], session['role'], q, None, None,
         datetime.now().strftime('%Y-%m-%d')),
    )
    conn.commit()
    new_id = cur.lastrowid
    row = conn.execute('SELECT * FROM qa_posts WHERE id=?', (new_id,)).fetchone()
    conn.close()
    return jsonify(dict(row))


@app.route('/api/qa/<int:qid>/answer', methods=['POST'])
@login_required
def post_answer(qid):
    if session.get('role') not in ('expert', 'admin'):
        return jsonify({"error": "Only experts can answer"}), 403
    data = request.json
    answer = data.get('answer', '').strip()
    conn = get_db()
    row = conn.execute('SELECT * FROM qa_posts WHERE id=?', (qid,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    conn.execute('UPDATE qa_posts SET answer=?, answered_by=? WHERE id=?',
                 (answer, session['name'], qid))
    conn.commit()
    updated = conn.execute('SELECT * FROM qa_posts WHERE id=?', (qid,)).fetchone()
    conn.close()
    # Notify the investor who asked, if we know who they are.
    if row['asker_email']:
        add_notification(row['asker_email'],
                         f"{session['name']} answered your question.", "answer")
    return jsonify(dict(updated))


@app.route('/api/qa/<int:qid>', methods=['DELETE'])
@login_required
def delete_qa(qid):
    if session.get('role') not in ('expert', 'admin'):
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    conn.execute('DELETE FROM qa_posts WHERE id=?', (qid,))
    conn.commit()
    conn.close()
    return jsonify({"message": "deleted"})


# ─── Feedback API (Module 9) ──────────────────────────────────────────────────
@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.json or {}
    category = data.get('category', 'General').strip()
    message  = data.get('message', '').strip()
    if not message:
        return jsonify({"error": "Message required"}), 400
    conn = get_db()
    conn.execute(
        '''INSERT INTO feedback (user_email, name, category, message, status, created_at)
           VALUES (?,?,?,?,?,?)''',
        (session['user'], session['name'], category, message, 'open',
         datetime.now().strftime('%Y-%m-%d %H:%M')),
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Thank you for your feedback!"})


@app.route('/api/feedback', methods=['GET'])
@login_required
def list_feedback():
    """Investors see their own feedback; admins see everything."""
    conn = get_db()
    if session.get('role') == 'admin':
        rows = conn.execute('SELECT * FROM feedback ORDER BY id DESC').fetchall()
    else:
        rows = conn.execute('SELECT * FROM feedback WHERE user_email=? ORDER BY id DESC',
                            (session['user'],)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/feedback/<int:fid>/respond', methods=['POST'])
@login_required
def respond_feedback(fid):
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    data = request.json or {}
    response = data.get('response', '').strip()
    conn = get_db()
    row = conn.execute('SELECT * FROM feedback WHERE id=?', (fid,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    conn.execute('UPDATE feedback SET admin_response=?, status=? WHERE id=?',
                 (response, 'resolved', fid))
    conn.commit()
    conn.close()
    if row['user_email']:
        add_notification(row['user_email'],
                         "An admin responded to your feedback.", "feedback")
    return jsonify({"message": "responded"})


# ─── Admin API ────────────────────────────────────────────────────────────────
@app.route('/api/admin/users', methods=['GET'])
@login_required
def admin_users():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    rows = conn.execute('SELECT email, name, role FROM users').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/users/<email>', methods=['DELETE'])
@login_required
def admin_delete_user(email):
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    if email == session['user']:
        return jsonify({"error": "Cannot delete yourself"}), 400
    conn = get_db()
    conn.execute('DELETE FROM users WHERE email=?', (email,))
    conn.commit()
    conn.close()
    return jsonify({"message": "deleted"})


if __name__ == '__main__':
    print(f"Training default model ({DEFAULT_TICKER}) on startup...")
    get_model(DEFAULT_TICKER)
    print("Model ready. Starting server...")
    app.run(debug=True, port=5000)
