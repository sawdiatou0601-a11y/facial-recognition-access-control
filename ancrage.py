"""Ancrage externe signe de la piste d'audit.

Principe : le journal SQLite reste sur la carte, mais une preuve signee de son etat
(dernier identifiant, nombre d'evenements, empreinte de tete) est ecrite hors du
repertoire du projet. Une troncature du journal devient alors detectable, puisque
l'ancrage conserve le nombre d'evenements attendu.
"""
import os, json, sqlite3, hashlib
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

DB_DEFAUT = "logs/audit_trail.db"
DOSSIER   = os.path.expanduser("~/ancrages")          # hors du repertoire du projet
CLE_PRIV  = os.path.expanduser("~/.ancrage_key")      # secret, permissions 600
CLE_PUB   = os.path.join(DOSSIER, "cle_publique.pem")
JOURNAL   = os.path.join(DOSSIER, "ancrages.jsonl")   # append-only

def _cles():
    os.makedirs(DOSSIER, exist_ok=True)
    if not os.path.exists(CLE_PRIV):
        priv = Ed25519PrivateKey.generate()
        with open(CLE_PRIV, "wb") as f:
            f.write(priv.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
        os.chmod(CLE_PRIV, 0o600)
        with open(CLE_PUB, "wb") as f:
            f.write(priv.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo))
        print(f"[ancrage] cle privee generee : {CLE_PRIV} (600)")
        print(f"[ancrage] cle publique       : {CLE_PUB}")
    priv = serialization.load_pem_private_key(open(CLE_PRIV, "rb").read(), password=None)
    pub  = serialization.load_pem_public_key(open(CLE_PUB, "rb").read())
    return priv, pub

def etat_journal(db=DB_DEFAUT):
    """Retourne (nombre d'evenements, dernier id, empreinte de tete)."""
    c = sqlite3.connect(db)
    n = c.execute("SELECT COUNT(*) FROM acces").fetchone()[0]
    r = c.execute("SELECT id, row_hash FROM acces ORDER BY id DESC LIMIT 1").fetchone()
    c.close()
    return (n, r[0], r[1]) if r else (0, 0, "")

def creer_ancrage(db=DB_DEFAUT, device_id=None):
    """Calcule l'etat du journal, le signe et l'ajoute au fichier d'ancrages."""
    priv, _ = _cles()
    n, dernier_id, dernier_hash = etat_journal(db)
    corps = {
        "device_id":  device_id or os.uname().nodename,
        "ts_utc":     datetime.now(timezone.utc).isoformat(),
        "count":      n,
        "last_id":    dernier_id,
        "last_hash":  dernier_hash,
    }
    charge = json.dumps(corps, sort_keys=True, separators=(",", ":")).encode()
    corps["signature"] = priv.sign(charge).hex()
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(corps, sort_keys=True) + "\n")
    return corps

def dernier_ancrage():
    if not os.path.exists(JOURNAL):
        return None
    lignes = [l for l in open(JOURNAL).read().splitlines() if l.strip()]
    return json.loads(lignes[-1]) if lignes else None

def verifier_ancrage(db=DB_DEFAUT):
    """Compare l'etat courant du journal au dernier ancrage signe.
    Retourne (ok, message)."""
    _, pub = _cles()
    anc = dernier_ancrage()
    if anc is None:
        return False, "Aucun ancrage enregistre"

    # 1. authenticite de l'ancrage lui-meme
    corps = {k: anc[k] for k in ("device_id", "ts_utc", "count", "last_id", "last_hash")}
    charge = json.dumps(corps, sort_keys=True, separators=(",", ":")).encode()
    try:
        pub.verify(bytes.fromhex(anc["signature"]), charge)
    except Exception:
        return False, "Ancrage falsifie : signature invalide"

    # 2. comparaison avec l'etat courant du journal
    n, dernier_id, dernier_hash = etat_journal(db)
    if n < anc["count"]:
        return False, (f"TRONCATURE detectee : {anc['count'] - n} evenement(s) manquant(s) "
                       f"(attendu {anc['count']}, trouve {n})")
    if n == anc["count"]:
        if dernier_id != anc["last_id"] or dernier_hash != anc["last_hash"]:
            return False, "Journal altere : l'empreinte de tete ne correspond pas a l'ancrage"
        return True, f"Journal conforme a l'ancrage ({n} evenements)"
    # n > count : des evenements ont ete ajoutes depuis l'ancrage
    c = sqlite3.connect(db)
    r = c.execute("SELECT row_hash FROM acces WHERE id = ?", (anc["last_id"],)).fetchone()
    c.close()
    if r is None:
        return False, f"Journal altere : l'evenement ancre (id {anc['last_id']}) a disparu"
    if r[0] != anc["last_hash"]:
        return False, f"Journal altere : l'evenement ancre (id {anc['last_id']}) a ete modifie"
    return True, (f"Journal conforme a l'ancrage, {n - anc['count']} evenement(s) ajoute(s) depuis")

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "verifier"
    if action == "creer":
        a = creer_ancrage()
        print(f"Ancrage cree : count={a['count']} last_id={a['last_id']} "
              f"hash={a['last_hash'][:16]}... ts={a['ts_utc']}")
        print(f"Enregistre dans : {JOURNAL}")
    else:
        ok, msg = verifier_ancrage()
        print(("OK  : " if ok else "ALERTE : ") + msg)
