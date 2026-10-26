import sqlite3, hashlib, os, csv
from datetime import datetime, timezone

DB_FILE = "logs/audit_trail.db"
GENESIS = "0" * 64

def _conn():
    os.makedirs("logs", exist_ok=True)
    c = sqlite3.connect(DB_FILE)
    c.execute("PRAGMA journal_mode=WAL;")
    return c

def init_db():
    c = _conn()
    c.executescript("""CREATE TABLE IF NOT EXISTS acces(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL, device_id TEXT NOT NULL, nom TEXT NOT NULL,
        decision TEXT NOT NULL, distance REAL, seuil REAL NOT NULL,
        modele TEXT NOT NULL, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL);""")
    c.commit(); c.close()

def _hash(ts, dev, nom, dec, dist, seuil, mod, prev):
    ds = "" if dist is None else f"{dist:.6f}"
    payload = "|".join([ts, dev, nom, dec, ds, f"{seuil:.6f}", mod, prev])
    return hashlib.sha256(payload.encode()).hexdigest()

def log_acces(nom, decision, distance, seuil=0.30, modele="Facenet512", device_id=None):
    device_id = device_id or os.uname().nodename
    ts = datetime.now(timezone.utc).isoformat()
    c = _conn()
    try:
        r = c.execute("SELECT row_hash FROM acces ORDER BY id DESC LIMIT 1").fetchone()
        prev = r[0] if r else GENESIS
        h = _hash(ts, device_id, nom, decision, distance, seuil, modele, prev)
        c.execute("""INSERT INTO acces(ts_utc,device_id,nom,decision,distance,seuil,modele,prev_hash,row_hash)
                     VALUES(?,?,?,?,?,?,?,?,?)""",
                  (ts, device_id, nom, decision, distance, seuil, modele, prev, h))
        c.commit()
    finally:
        c.close()

def verifier_chaine():
    c = _conn()
    try:
        rows = c.execute("""SELECT id,ts_utc,device_id,nom,decision,distance,seuil,modele,prev_hash,row_hash
                            FROM acces ORDER BY id""").fetchall()
    finally:
        c.close()
    prev = GENESIS
    for r in rows:
        rid, ts, dev, nom, dec, dist, seuil, mod, ph, rh = r
        if ph != prev:
            return False, f"Chaine rompue a l'id {rid}"
        if _hash(ts, dev, nom, dec, dist, seuil, mod, ph) != rh:
            return False, f"Ligne alteree a l'id {rid}"
        prev = rh
    return True, f"Chaine valide : {len(rows)} evenements"

def exporter_csv(chemin="logs/audit_export.csv"):
    c = _conn()
    try:
        rows = c.execute("SELECT * FROM acces ORDER BY id").fetchall()
        cols = [d[0] for d in c.execute("SELECT * FROM acces LIMIT 0").description]
    finally:
        c.close()
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    return chemin

if __name__ == "__main__":
    init_db()
    print(verifier_chaine()[1])
    print("Base :", DB_FILE)
