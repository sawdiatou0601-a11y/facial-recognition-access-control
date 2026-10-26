import os, glob, pickle
import numpy as np, cv2
from deepface import DeepFace

MODEL="Facenet512"; DET="retinaface"; HAUT=0.45; DATASET="dataset"
def perioc(img_path):
    try:
        faces=DeepFace.extract_faces(img_path=img_path, detector_backend=DET, enforce_detection=True, align=False)
    except Exception: return None
    fa=faces[0]["facial_area"]; x,y,w,h=fa["x"],fa["y"],fa["w"],fa["h"]
    img=cv2.imread(img_path); crop=img[max(0,y):max(0,y+int(HAUT*h)), max(0,x):x+w]
    if crop.size==0: return None
    try:
        rep=DeepFace.represent(img_path=crop, model_name=MODEL, detector_backend="skip", enforce_detection=False, align=False)
        return np.array(rep[0]["embedding"])
    except Exception: return None

emb=[]; noms=[]; ok=0; skip=0
for personne in sorted(os.listdir(DATASET)):
    dd=os.path.join(DATASET,personne)
    if not os.path.isdir(dd): continue
    imgs=sorted(glob.glob(os.path.join(dd,"*.jpg")))
    print(f"=== {personne} ({len(imgs)}) ===", flush=True)
    for img in imgs:
        e=perioc(img)
        if e is None: skip+=1; continue
        emb.append(e); noms.append(personne); ok+=1
pickle.dump({"embeddings":np.array(emb),"noms":noms}, open("encodings/perioculaire.pkl","wb"))
print(f"\nPeri-oculaire encode: {ok}  ignore: {skip}  total: {len(noms)}", flush=True)
