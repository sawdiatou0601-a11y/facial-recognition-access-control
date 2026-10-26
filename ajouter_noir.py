import glob
import pickle
import numpy as np
from deepface import DeepFace

PKL = "encodings/encodings.pkl"
with open(PKL, "rb") as f:
    data = pickle.load(f)
emb = list(data["embeddings"])
noms = list(data["noms"])
avant = len(noms)

def encode(img):
    for backend in ("opencv", "ssd"):
        try:
            rep = DeepFace.represent(img_path=img, model_name="Facenet512",
                                     enforce_detection=True, detector_backend=backend)
            return rep[0]["embedding"], backend
        except Exception:
            continue
    return None, None

for img in sorted(glob.glob("dataset/Sawdiatou/noir_*.jpg")):
    e, b = encode(img)
    if e is None:
        print("SKIP", img)
    else:
        emb.append(e)
        noms.append("Sawdiatou")
        print("OK  ", img, "(", b, ")")

with open(PKL, "wb") as f:
    pickle.dump({"embeddings": np.array(emb), "noms": noms}, f)
print(f"\nAjoutes: {len(noms) - avant}   Total: {len(noms)}")
