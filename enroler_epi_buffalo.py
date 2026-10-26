import sys, os, glob, pickle, shutil
import numpy as np, cv2
from insightface.app import FaceAnalysis

dossier = sys.argv[1]           # ex: test_occlusion/nadine_enrol
personne = sys.argv[2]          # ex: Nadine
PKL = "encodings/buffalo.pkl"

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))

shutil.copy(PKL, PKL.replace(".pkl", "_backup.pkl"))
d = pickle.load(open(PKL, "rb"))
E = list(d["embeddings"]); N = list(d["noms"]); avant = len(N)

dest = os.path.join("dataset", personne)
os.makedirs(dest, exist_ok=True)

for img in sorted(glob.glob(os.path.join(dossier, "*.jpg"))):
    frame = cv2.imread(img)
    faces = app.get(frame)
    if not faces:
        print("SKIP", os.path.basename(img)); continue
    f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    E.append(np.array(f.normed_embedding, float)); N.append(personne)
    shutil.copy(img, os.path.join(dest, "epi_" + os.path.basename(img)))
    print("OK  ", os.path.basename(img))

pickle.dump({"embeddings": np.array(E), "noms": N}, open(PKL, "wb"))
print(f"\nAjoutes: {len(N)-avant}   Total galerie: {len(N)}")
