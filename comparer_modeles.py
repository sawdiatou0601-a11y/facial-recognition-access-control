import os, glob, time
import numpy as np
from deepface import DeepFace

DATASET = "dataset"
MODELES = ["Facenet512", "ArcFace", "GhostFaceNet"]
DETECTOR = "ssd"

def lister():
    items = []
    for personne in sorted(os.listdir(DATASET)):
        d = os.path.join(DATASET, personne)
        if not os.path.isdir(d):
            continue
        for img in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            items.append((personne, img))
    return items

def encoder(items, modele):
    embs, noms, temps = [], [], []
    for personne, img in items:
        t0 = time.perf_counter()
        try:
            rep = DeepFace.represent(img_path=img, model_name=modele,
                                     enforce_detection=True, detector_backend=DETECTOR)
            emb = rep[0]["embedding"]
        except Exception:
            continue
        temps.append(time.perf_counter() - t0)
        embs.append(emb); noms.append(personne)
    return np.array(embs, float), np.array(noms), np.array(temps)

def evaluer(embs, noms):
    X = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    D = 1 - X @ X.T
    np.fill_diagonal(D, np.inf)
    n = len(noms)
    nn = np.argmin(D, axis=1)
    rank1 = float(np.mean(noms[nn] == noms))
    g, i = [], []
    for k in range(n):
        s = (noms == noms[k]).copy(); s[k] = False; f = noms != noms[k]
        if s.any(): g.append(D[k, s].min())
        if f.any(): i.append(D[k, f].min())
    g, i = np.array(g), np.array(i)
    ts = np.linspace(0, 2, 2001)
    FAR = np.array([(i <= t).mean() for t in ts]); FRR = np.array([(g > t).mean() for t in ts])
    idx = int(np.argmin(np.abs(FAR - FRR)))
    return rank1, (FAR[idx] + FRR[idx]) / 2, ts[idx]

def main():
    items = lister()
    print(f"{len(items)} images, {len(MODELES)} modeles a comparer\n", flush=True)
    resultats = []
    for m in MODELES:
        print(f"== {m} : chauffe + encodage... ==", flush=True)
        try:
            DeepFace.represent(img_path=items[0][1], model_name=m,
                               enforce_detection=False, detector_backend=DETECTOR)
        except Exception as e:
            print(f"   (chauffe: {e})", flush=True)
        embs, noms, temps = encoder(items, m)
        if len(embs) < 5:
            print(f"   {m} : trop peu de visages encodes, ignore.", flush=True); continue
        rank1, eer, t_eer = evaluer(embs, noms)
        lat = float(temps.mean()) * 1000
        resultats.append((m, embs.shape[1], lat, rank1 * 100, eer * 100, t_eer))
        print(f"   OK : dim={embs.shape[1]} latence={lat:.0f}ms rang1={rank1*100:.1f}% EER={eer*100:.1f}%\n", flush=True)

    print("\n================ TABLEAU COMPARATIF ================")
    print(f"{'Modele':12} | {'dim':4} | {'latence ms':10} | {'rang-1 %':8} | {'EER %':6} | seuil")
    for m, dim, lat, r1, eer, t in resultats:
        print(f"{m:12} | {dim:4d} | {lat:10.0f} | {r1:8.1f} | {eer:6.1f} | {t:.2f}")

if __name__ == "__main__":
    main()
