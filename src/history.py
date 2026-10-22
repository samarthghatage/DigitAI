import sqlite3
from datetime import datetime,timezone
from .config import HISTORY_DB
def connect():
    c=sqlite3.connect(HISTORY_DB); c.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,model TEXT NOT NULL,predicted_digit INTEGER NOT NULL,confidence REAL NOT NULL,inference_ms REAL NOT NULL,source TEXT NOT NULL)"); c.commit(); return c
def add(model,digit,confidence,inference_ms,source):
    with connect() as c: c.execute("INSERT INTO predictions(timestamp,model,predicted_digit,confidence,inference_ms,source) VALUES(?,?,?,?,?,?)",(datetime.now(timezone.utc).isoformat(),model,int(digit),float(confidence),float(inference_ms),source))
def recent(limit=30):
    with connect() as c: return c.execute("SELECT id,timestamp,model,predicted_digit,confidence,inference_ms,source FROM predictions ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
def clear():
    with connect() as c: c.execute("DELETE FROM predictions")
