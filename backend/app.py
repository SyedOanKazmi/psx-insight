from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from functools import wraps
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.secret_key = secrets.token_hex(32)

# ─── In-memory user store (replace with DB later) ────────────────────────────
USERS = {
    "admin@psx.com":  {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin",    "name": "Admin"},
    "expert@psx.com": {"password": hashlib.sha256("expert123".encode()).hexdigest(), "role": "expert",   "name": "Dr. Ayesha Khan"},
    "user@psx.com":   {"password": hashlib.sha256("user123".encode()).hexdigest(),   "role": "investor", "name": "Ali Raza"},
}

# ─── In-memory Q&A store ──────────────────────────────────────────────────────
QA_POSTS = [
    {"id": 1, "author": "Ali Raza", "role": "investor", "question": "Is OGDC a good long-term investment?",
     "answer": "OGDC has strong fundamentals with consistent dividend payouts. However, watch for oil price volatility.", "answered_by": "Dr. Ayesha Khan", "date": "2024-12-01"},
    {"id": 2, "author": "Sara Ahmed", "role": "investor", "question": "What is the outlook for PSX banking sector in 2025?",
     "answer": None, "answered_by": None, "date": "2024-12-10"},
]
qa_counter = 3

# ─── Stock data & model cache ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/PSX_KSE100.csv')
model_cache = {}

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

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
def load_stock_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

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

def get_model():
    if 'model' not in model_cache:
        df = load_stock_data()
        df_feat = engineer_features(df)
        model, features, mape, X_test, y_test, y_pred = train_model(df_feat)
        model_cache['model']    = model
        model_cache['features'] = features
        model_cache['mape']     = mape
        model_cache['df']       = df
        model_cache['df_feat']  = df_feat
        model_cache['X_test']   = X_test
        model_cache['y_test']   = y_test.values
        model_cache['y_pred']   = y_pred
    return model_cache

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

@app.route('/qa')
@login_required
def qa_page():
    return render_template('qa.html', user=session['name'], role=session['role'])

@app.route('/admin')
@login_required
def admin_page():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    return render_template('admin.html', user=session['name'], role=session['role'], users=USERS)

# ─── Auth API ─────────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email', '').strip()
    pw    = data.get('password', '')
    user  = USERS.get(email)
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
    if email in USERS:
        return jsonify({"error": "Email already registered"}), 409
    USERS[email] = {"password": hash_pw(pw), "role": "investor", "name": name}
    session['user'] = email
    session['role'] = 'investor'
    session['name'] = name
    return jsonify({"message": "ok", "role": "investor", "name": name})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({"message": "ok"})

# ─── Stock Data API ───────────────────────────────────────────────────────────
@app.route('/api/stock/history')
@login_required
def stock_history():
    df = load_stock_data()
    recent = df.tail(365)
    return jsonify({
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
    df = load_stock_data()
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    change = round(latest['Close'] - prev['Close'], 2)
    pct    = round((change / prev['Close']) * 100, 2)
    return jsonify({
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
    days = int(request.args.get('days', 30))
    days = min(max(days, 7), 90)
    cache = get_model()
    future = predict_future(cache['model'], cache['df_feat'], cache['features'], days)
    accuracy = round((1 - cache['mape']) * 100, 2)

    # historical actuals vs predictions on test set
    test_dates = cache['df_feat'].iloc[-len(cache['y_test']):]['Date'].dt.strftime('%Y-%m-%d').tolist()
    return jsonify({
        "accuracy":       accuracy,
        "future":         future,
        "test_dates":     test_dates[-60:],
        "test_actual":    [round(v,2) for v in cache['y_test'][-60:]],
        "test_predicted": [round(v,2) for v in cache['y_pred'][-60:]],
    })

# ─── Q&A API ──────────────────────────────────────────────────────────────────
@app.route('/api/qa', methods=['GET'])
@login_required
def get_qa():
    return jsonify(QA_POSTS)

@app.route('/api/qa', methods=['POST'])
@login_required
def post_question():
    global qa_counter
    data = request.json
    q = data.get('question', '').strip()
    if not q:
        return jsonify({"error": "Question required"}), 400
    post = {
        "id": qa_counter, "author": session['name'], "role": session['role'],
        "question": q, "answer": None, "answered_by": None,
        "date": datetime.now().strftime('%Y-%m-%d')
    }
    QA_POSTS.append(post)
    qa_counter += 1
    return jsonify(post)

@app.route('/api/qa/<int:qid>/answer', methods=['POST'])
@login_required
def post_answer(qid):
    if session.get('role') not in ('expert', 'admin'):
        return jsonify({"error": "Only experts can answer"}), 403
    data = request.json
    answer = data.get('answer', '').strip()
    for post in QA_POSTS:
        if post['id'] == qid:
            post['answer']      = answer
            post['answered_by'] = session['name']
            return jsonify(post)
    return jsonify({"error": "Not found"}), 404

@app.route('/api/qa/<int:qid>', methods=['DELETE'])
@login_required
def delete_qa(qid):
    global QA_POSTS
    if session.get('role') not in ('expert', 'admin'):
        return jsonify({"error": "Unauthorized"}), 403
    QA_POSTS = [p for p in QA_POSTS if p['id'] != qid]
    return jsonify({"message": "deleted"})

# ─── Admin API ────────────────────────────────────────────────────────────────
@app.route('/api/admin/users', methods=['GET'])
@login_required
def admin_users():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    users = [{"email": e, "name": v['name'], "role": v['role']} for e, v in USERS.items()]
    return jsonify(users)

@app.route('/api/admin/users/<email>', methods=['DELETE'])
@login_required
def admin_delete_user(email):
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    if email == session['user']:
        return jsonify({"error": "Cannot delete yourself"}), 400
    USERS.pop(email, None)
    return jsonify({"message": "deleted"})

if __name__ == '__main__':
    print("Training model on startup...")
    get_model()
    print("Model ready. Starting server...")
    app.run(debug=True, port=5000)
