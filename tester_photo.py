import sys
import pickle
import numpy as np
from deepface import DeepFace

image = sys.argv[1] if len(sys.argv) > 1 else "capture_test.jpg"
backend = sys.argv[2] if len(sys.argv) > 2 else "opencv"

with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)
embeddings = np.array(data["embeddings"])
noms = data["noms"]

def dcos(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Image={image}  detecteur={backend}")
try:
    rep = DeepFace.represent(img_path=image, model_name="Facenet512",
                             enforce_detection=True, detector_backend=backend)
    e = np.array(rep[0]["embedding"])
    d = np.array([dcos(e, r) for r in embeddings])
    i = int(np.argmin(d))
    print(f"VISAGE DETECTE -> {noms[i]}  distance={d[i]:.4f}")
except Exception as ex:
    print("PAS DE VISAGE:", ex)
