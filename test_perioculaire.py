import sys, os, glob
import numpy as np, cv2
from deepface import DeepFace

MODEL = "Facenet512"; DET = "retinaface"; HAUT = 0.45

def perioc(img_path):
    try:
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend=DET, enforce_detection=True, align=False)
    except Exception:
        return None
    fa = faces[0]["facial_area"]; x,y,w,h = fa["x"],fa["y"],fa["w"],fa["h"]
    img = cv2.imread(img_path)
    crop = img[max(0,y):max(0,y+int(HAUT*h)), max(0,x):x+w]
    if crop.size == 0: return None
    try:
        rep = DeepFace.represent(img_path=crop, model_name=MODEL, detector_backend="skip", enforce_detection=False, align=False)
        return np.array(rep[0]["embedding"])
    except Exception:
        return None

def dcos(a,b): return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

base = []
for p in sorted(glob.glob("test_occlusion/masque_enrol/*.jpg")):
    e = perioc(p)
    if e is not None: base.append(e)
print(f"Base peri-oculaire Sawdiatou : {len(base)} empreintes\n")

def tester(dossier, label):
    print(f"=== {label} ===")
    ds = []
    for p in sorted(glob.glob(os.path.join(dossier, "*.jpg"))):
        e = perioc(p)
        if e is None:
            print(f"  {os.path.basename(p)} : pas de visage"); continue
        dmin = min(dcos(e, b) for b in base)
        ds.append(dmin)
        print(f"  {os.path.basename(p)} : distance min vers Sawdiatou = {dmin:.3f}")
    if ds: print(f"  --> moyenne = {np.mean(ds):.3f}\n")

tester("test_occlusion/masque_seul", "Sawdiatou masquee (doit etre PROCHE)")
tester("test_occlusion/nadine_masque", "Nadine masquee (doit etre LOIN)")
