import sys, glob, os
import numpy as np, cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

def emb_de(path):
    img = cv2.imread(path)
    faces = app.get(img)
    if not faces: return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = f.normed_embedding
    return np.array(e, dtype=float)

def dcos(a,b): return 1 - float(np.dot(a,b))

# Galerie : toutes les photos du dataset
gal_emb, gal_nom = [], []
for personne in sorted(os.listdir("dataset")):
    d = os.path.join("dataset", personne)
    if not os.path.isdir(d): continue
    n = 0
    for img in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        e = emb_de(img)
        if e is None: continue
        gal_emb.append(e); gal_nom.append(personne); n += 1
    print(f"  {personne}: {n} encodees", flush=True)
gal_emb = np.array(gal_emb); gal_nom = np.array(gal_nom)
print(f"\nGalerie buffalo_m : {len(gal_nom)} empreintes\n", flush=True)

def tester(dossier, attendu):
    print(f"=== {dossier} (attendu: {attendu}) ===")
    bons = 0; tot = 0
    for p in sorted(glob.glob(os.path.join(dossier, "*.jpg"))):
        e = emb_de(p)
        if e is None:
            print(f"  {os.path.basename(p)}: pas de visage detecte"); continue
        dd = np.array([dcos(e, g) for g in gal_emb]); i = int(np.argmin(dd))
        tot += 1
        if gal_nom[i] == attendu: bons += 1
        print(f"  {os.path.basename(p)}: {gal_nom[i]} (d={dd[i]:.3f})")
    if tot: print(f"  --> correct: {bons}/{tot}\n")

tester("test_occlusion/masque_seul", "Sawdiatou")
tester("test_occlusion/nadine_masque", "Nadine")
