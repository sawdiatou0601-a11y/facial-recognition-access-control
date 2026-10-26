import sys, os, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
sujet   = sys.argv[1]                                  # personne figurant sur le support
support = sys.argv[2]                                  # photo / video / reel
N       = int(sys.argv[3]) if len(sys.argv) > 3 else 10

d = pickle.load(open("encodings/buffalo.pkl", "rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(f">>> Support : {support} | sujet represente : {sujet} | {N} presentations")
print(">>> Presente le support devant la camera. Debut dans 5 s..."); time.sleep(5)

os.makedirs("logs", exist_ok=True)
p = "logs/mesures_presentation.csv"; neuf = not os.path.exists(p)
f = open(p, "a", newline=""); w = csv.writer(f)
if neuf: w.writerow(["horodatage","sujet","support","predit","distance","decision","detecte"])

k=0; essais=0; accepte=0; non_det=0; dists=[]
while k < N and essais < N*8:
    essais += 1
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        non_det += 1; time.sleep(0.4); continue
    fc = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = np.array(fc.normed_embedding, float)
    dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
    ok = dist <= SEUIL
    predit = NOMS[i] if ok else "INCONNU"
    if ok: accepte += 1
    dists.append(dist); k += 1
    w.writerow([time.strftime("%H:%M:%S"), sujet, support, predit, f"{dist:.4f}",
                "ACCEPTE" if ok else "REFUSE", 1])
    marque = "  <-- USURPATION REUSSIE" if ok else ""
    print(f"  {k:2d}/{N}: {predit:12} d={dist:.3f} {'ACCEPTE' if ok else 'REFUSE '}{marque}")
    time.sleep(0.6)
f.close(); cam.stop()

dists = np.array(dists)
print(f"\n=== {support.upper()} de {sujet} ===")
print(f"Presentations detectees : {k}   (non detectees : {non_det})")
print(f"Acceptations            : {accepte}   soit {accepte/max(k,1)*100:.1f} %")
print(f"Distance moyenne        : {dists.mean():.3f}  (min {dists.min():.3f} / max {dists.max():.3f})")
