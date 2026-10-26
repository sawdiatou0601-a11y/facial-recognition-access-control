import shutil, sqlite3, json, os
import db_audit, ancrage

ORIG = "logs/audit_trail.db"
resultats = []

def controle(chemin, libelle):
    db_audit.DB_FILE = chemin
    ok_chaine, msg_chaine = db_audit.verifier_chaine()
    ok_ancre, msg_ancre   = ancrage.verifier_ancrage(chemin)
    resultats.append((libelle, ok_chaine, msg_chaine, ok_ancre, msg_ancre))
    print(f"\n{libelle}")
    print(f"   chainage local  : {'VALIDE ' if ok_chaine else 'ALERTE '} | {msg_chaine}")
    print(f"   ancrage externe : {'VALIDE ' if ok_ancre else 'ALERTE '} | {msg_ancre}")

c = sqlite3.connect(ORIG)
ids = [r[0] for r in c.execute("SELECT id FROM acces ORDER BY id").fetchall()]
c.close()
cible = ids[len(ids)//2]; dernier = ids[-1]
print(f"Base originale : {len(ids)} enregistrements | cible intermediaire id={cible} | dernier id={dernier}")

controle(ORIG, "ETAT DE REFERENCE (aucune manipulation)")

shutil.copy(ORIG, "logs/a1.db")
c = sqlite3.connect("logs/a1.db"); c.execute("DELETE FROM acces WHERE id=?", (cible,)); c.commit(); c.close()
controle("logs/a1.db", f"ATTAQUE 1 — suppression de l'enregistrement intermediaire (id {cible})")

shutil.copy(ORIG, "logs/a2.db")
c = sqlite3.connect("logs/a2.db"); c.execute("DELETE FROM acces WHERE id=?", (dernier,)); c.commit(); c.close()
controle("logs/a2.db", f"ATTAQUE 2 — suppression du dernier enregistrement (id {dernier})")

shutil.copy(ORIG, "logs/a3.db")
c = sqlite3.connect("logs/a3.db")
c.execute("UPDATE acces SET ts_utc='2020-01-01T00:00:00+00:00' WHERE id=?", (cible,)); c.commit(); c.close()
controle("logs/a3.db", f"ATTAQUE 3 — modification de l'horodatage (id {cible})")

shutil.copy(ORIG, "logs/a4.db")
c = sqlite3.connect("logs/a4.db")
c.execute("UPDATE acces SET decision='AUTORISE', distance=0.0001 WHERE id=?", (cible,)); c.commit(); c.close()
controle("logs/a4.db", f"ATTAQUE 4 — modification de la decision et de la distance (id {cible})")

shutil.copy(ORIG, "logs/a5.db")
c = sqlite3.connect("logs/a5.db")
c.execute("DELETE FROM acces WHERE id >= ?", (dernier-4,)); c.commit(); c.close()
controle("logs/a5.db", f"ATTAQUE 5 — troncature des 5 derniers enregistrements")

print("\n" + "="*100)
print("ATTAQUE 6 — falsification de l'ancrage lui-meme")
shutil.copy(ancrage.JOURNAL, "/tmp/ancrages_sauve.jsonl")
lignes = [l for l in open(ancrage.JOURNAL).read().splitlines() if l.strip()]
faux = json.loads(lignes[-1]); faux["count"] = faux["count"] - 5      # l'attaquant abaisse le compteur
lignes[-1] = json.dumps(faux, sort_keys=True)
open(ancrage.JOURNAL, "w").write("\n".join(lignes) + "\n")
ok, msg = ancrage.verifier_ancrage("logs/a5.db")
print(f"   ancrage externe : {'VALIDE ' if ok else 'ALERTE '} | {msg}")
resultats.append(("ATTAQUE 6 — falsification de l'ancrage", None, "-", ok, msg))
shutil.copy("/tmp/ancrages_sauve.jsonl", ancrage.JOURNAL)
print("   (fichier d'ancrages restaure)")

print("\n" + "="*100)
print(f"{'Manipulation':58} | {'chainage local':>15} | {'ancrage externe':>16}")
print("-"*100)
for lib, okc, _, oka, _ in resultats:
    c1 = "-" if okc is None else ("non detectee" if okc else "DETECTEE")
    c2 = "non detectee" if oka else "DETECTEE"
    print(f"{lib[:58]:58} | {c1:>15} | {c2:>16}")
print("="*100)
print("La base originale et le fichier d'ancrages n'ont pas ete modifies.")
