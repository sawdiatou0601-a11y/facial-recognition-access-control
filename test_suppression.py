import shutil, sqlite3, importlib
import db_audit

ORIG = "logs/audit_trail.db"

def etat(chemin, titre):
    db_audit.DB_FILE = chemin
    ok, msg = db_audit.verifier_chaine()
    c = sqlite3.connect(chemin)
    n = c.execute("SELECT COUNT(*) FROM acces").fetchone()[0]
    ids = [r[0] for r in c.execute("SELECT id FROM acces ORDER BY id").fetchall()]
    c.close()
    print(f"{titre:34} | {n:2d} lignes | ids {ids} | {'VALIDE' if ok else 'ALTEREE'} | {msg}")
    return ok

print("=" * 110)
db_audit.DB_FILE = ORIG
c = sqlite3.connect(ORIG)
total = c.execute("SELECT COUNT(*) FROM acces").fetchone()[0]
ids = [r[0] for r in c.execute("SELECT id FROM acces ORDER BY id").fetchall()]
c.close()
print(f"Base originale : {total} enregistrements (ids {ids[:12]}{'...' if len(ids)>12 else ''})")
print("=" * 110)

# --- CAS 1 : suppression d'un enregistrement intermediaire ---
print("\n--- CAS 1 : suppression d'un enregistrement INTERMEDIAIRE ---")
cible = ids[len(ids)//2]
shutil.copy(ORIG, "logs/test_sup_milieu.db")
etat("logs/test_sup_milieu.db", "Avant suppression")
c = sqlite3.connect("logs/test_sup_milieu.db")
c.execute("DELETE FROM acces WHERE id = ?", (cible,)); c.commit(); c.close()
print(f"  -> enregistrement id={cible} supprime")
etat("logs/test_sup_milieu.db", "Apres suppression (milieu)")

# --- CAS 2 : suppression du dernier enregistrement ---
print("\n--- CAS 2 : suppression du DERNIER enregistrement ---")
dernier = ids[-1]
shutil.copy(ORIG, "logs/test_sup_fin.db")
etat("logs/test_sup_fin.db", "Avant suppression")
c = sqlite3.connect("logs/test_sup_fin.db")
c.execute("DELETE FROM acces WHERE id = ?", (dernier,)); c.commit(); c.close()
print(f"  -> enregistrement id={dernier} supprime")
etat("logs/test_sup_fin.db", "Apres suppression (fin)")

# --- CAS 3 : modification d'un horodatage ---
print("\n--- CAS 3 : modification d'un HORODATAGE ---")
shutil.copy(ORIG, "logs/test_mod_ts.db")
etat("logs/test_mod_ts.db", "Avant modification")
c = sqlite3.connect("logs/test_mod_ts.db")
c.execute("UPDATE acces SET ts_utc='2020-01-01T00:00:00+00:00' WHERE id = ?", (cible,))
c.commit(); c.close()
print(f"  -> horodatage de l'id={cible} remplace")
etat("logs/test_mod_ts.db", "Apres modification horodatage")

# --- CAS 4 : modification d'une distance ---
print("\n--- CAS 4 : modification d'une DISTANCE ---")
shutil.copy(ORIG, "logs/test_mod_dist.db")
etat("logs/test_mod_dist.db", "Avant modification")
c = sqlite3.connect("logs/test_mod_dist.db")
c.execute("UPDATE acces SET distance=0.0001 WHERE id = ?", (cible,)); c.commit(); c.close()
print(f"  -> distance de l'id={cible} remplacee par 0.0001")
etat("logs/test_mod_dist.db", "Apres modification distance")

print("\n" + "=" * 110)
print("La base originale n'a jamais ete modifiee : toutes les manipulations ont porte sur des copies.")
