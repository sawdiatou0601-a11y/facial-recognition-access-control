import sys, glob, os, pickle
import numpy as np, cv2
from deepface import DeepFace

MODEL="Facenet512"; DET="retinaface"; HAUT=0.45; SEUIL=0.32
d=pickle.load(open("encodings/perioculaire.pkl","rb"))
refs=np.array(d["embeddings"],float); noms=np.array(d["noms"])
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
def dcos(a,b): return 1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
dossier=sys.argv[1] if len(sys.argv)>1 else "test_occlusion/masque_seul"
print(f"=== {dossier} (seuil {SEUIL}) ===")
for p in sorted(glob.glob(os.path.join(dossier,"*.jpg"))):
    e=perioc(p)
    if e is None: print(f"  {os.path.basename(p)}: pas de visage"); continue
    dd=np.array([dcos(e,r) for r in refs]); i=int(np.argmin(dd)); dist=float(dd[i])
    dec=noms[i] if dist<=SEUIL else "INCONNU"
    print(f"  {os.path.basename(p)}: {dec} (d={dist:.3f})")
