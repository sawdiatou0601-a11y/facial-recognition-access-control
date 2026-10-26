import os, glob, pickle
import numpy as np, cv2
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
E, N = [], []
for personne in sorted(os.listdir("dataset")):
    d = os.path.join("dataset", personne)
    if not os.path.isdir(d): continue
    for img in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        f = app.get(cv2.imread(img))
        if not f: continue
        E.append(np.array(f[0].normed_embedding, float)); N.append(personne)
    print("fait:", personne, flush=True)
pickle.dump({"embeddings": np.array(E), "noms": N}, open("encodings/buffalo.pkl", "wb"))
print(f"Galerie buffalo_l : {len(N)} empreintes", flush=True)
