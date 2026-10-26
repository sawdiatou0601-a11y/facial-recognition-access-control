import glob, pickle, shutil, os
import numpy as np
from deepface import DeepFace

PKL = "encodings/encodings.pkl"
DOSSIER = "test_occlusion/masque_enrol"
NOM = "Sawdiatou"
DETECTEUR = "retinaface"

shutil.copy(PKL, "encodings/encodings_avant_epi.pkl")  # sauvegarde
data = pickle.load(open(PKL, "rb"))
emb = list(data["embeddings"]); noms = list(data["noms"])
avant = len(noms)

for img in sorted(glob.glob(os.path.join(DOSSIER, "*.jpg"))):
    try:
        rep = DeepFace.represent(img_path=img, model_name="Facenet512",
                                 enforce_detection=True, detector_backend=DETECTEUR)
        emb.append(rep[0]["embedding"]); noms.append(NOM)
        print("OK  ", os.path.basename(img))
    except Exception as e:
        print("SKIP", os.path.basename(img), "->", str(e)[:50])

pickle.dump({"embeddings": np.array(emb), "noms": noms}, open(PKL, "wb"))
print(f"\nAjoutes: {len(noms)-avant}   Total: {len(noms)}")
