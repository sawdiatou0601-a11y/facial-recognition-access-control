import time, pickle
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

N = 15
d = pickle.load(open("encodings/buffalo.pkl","rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)

print(">>> Chauffe du modele...")
for _ in range(3):
    app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(">>> Place-toi face a la camera. Mesure dans 4 s..."); time.sleep(4)

t_acq, t_ia, t_cmp = [], [], []
faits = 0; essais = 0
while faits < N and essais < N*8:
    essais += 1
    a = time.perf_counter()
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    b = time.perf_counter()
    faces = app.get(bgr)
    c = time.perf_counter()
    if not faces: continue
    f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = np.array(f.normed_embedding, float)
    dd = 1 - G @ e; int(np.argmin(dd))
    z = time.perf_counter()
    t_acq.append((b-a)*1000); t_ia.append((c-b)*1000); t_cmp.append((z-c)*1000)
    faits += 1
    print(f"  mesure {faits}/{N}")
cam.stop()

def stat(nom, arr):
    arr = np.array(arr)
    print(f"{nom:42} moyenne={arr.mean():7.1f} ms   ecart-type={arr.std():6.1f} ms")

if faits:
    print(f"\n--- Temps de traitement buffalo_l @1280x960 ({faits} mesures) ---")
    stat("Acquisition image", t_acq)
    stat("Detection + reconnaissance (buffalo_l)", t_ia)
    stat("Comparaison a la galerie + decision", t_cmp)
    stat("TOTAL bout-en-bout", np.array(t_acq)+np.array(t_ia)+np.array(t_cmp))
else:
    print("Aucun visage detecte.")
