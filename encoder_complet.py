import os
import glob
import pickle
import numpy as np
from deepface import DeepFace

DATASET = "dataset"
emb, noms = [], []
ok = skip = 0

for personne in sorted(os.listdir(DATASET)):
    d = os.path.join(DATASET, personne)
    if not os.path.isdir(d):
        continue
    imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
    print(f"=== {personne} ({len(imgs)} images) ===", flush=True)
    for img in imgs:
        found = None
        for backend in ("opencv", "ssd"):
            try:
                rep = DeepFace.represent(img_path=img, model_name="Facenet512",
                                         enforce_detection=True, detector_backend=backend)
                found = rep[0]["embedding"]
                break
            except Exception:
                continue
        if found is None:
            skip += 1
            print("  SKIP", os.path.basename(img), flush=True)
            continue
        emb.append(found)
        noms.append(personne)
        ok += 1
        print("  OK  ", os.path.basename(img), flush=True)

with open("encodings/encodings.pkl", "wb") as f:
    pickle.dump({"embeddings": np.array(emb), "noms": noms}, f)
print(f"\nEncodes: {ok}   Ignores: {skip}   Total: {len(noms)}", flush=True)
