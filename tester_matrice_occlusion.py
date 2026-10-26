import sys, os, glob, pickle
import numpy as np
from deepface import DeepFace

SEUIL = 0.30
MODELE = "Facenet512"
DETECTEURS = ["retinaface"]
ATTENDUE = "Sawdiatou"

dossier = sys.argv[1] if len(sys.argv) > 1 else "test_occlusion/masque"
photos = sorted(glob.glob(os.path.join(dossier, "*.jpg")))
d = pickle.load(open("encodings/encodings.pkl", "rb"))
refs = np.array(d["embeddings"], float); noms = np.array(d["noms"])

def dcos(a, b): return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Dossier: {dossier} ({len(photos)} photos) | Modele: {MODELE} | base SANS EPI\n", flush=True)
print(f"{'Detecteur':12} | {'detectes':9} | {'reconnus':9} | {'dist.moy':8}")
for det in DETECTEURS:
    print(f"  ... {det} (1er lancement = telechargement du modele) ...", flush=True)
    detectes = 0; reconnus = 0; dists = []
    for p in photos:
        try:
            rep = DeepFace.represent(img_path=p, model_name=MODELE,
                                     enforce_detection=True, detector_backend=det)
            emb = np.array(rep[0]["embedding"])
        except Exception:
            continue
        detectes += 1
        dd = np.array([dcos(emb, r) for r in refs]); i = int(np.argmin(dd))
        dist = float(dd[i]); dists.append(dist)
        if noms[i] == ATTENDUE and dist <= SEUIL:
            reconnus += 1
    dm = f"{np.mean(dists):.3f}" if dists else "-"
    print(f"{det:12} | {detectes:>3}/{len(photos):<5} | {reconnus:>3}/{len(photos):<5} | {dm:>8}", flush=True)
