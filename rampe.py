import os, sys, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
condition = sys.argv[1] if len(sys.argv) > 1 else "rampe"
d = pickle.load(open("encodings/buffalo.pkl","rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

os.makedirs("logs", exist_ok=True)
p = "logs/mesures_eclairage.csv"; neuf = not os.path.exists(p)
f = open(p, "a", newline=""); w = csv.writer(f)
if neuf: w.writerow(["horodatage","condition","predit","distance","luminance","latence_ms","decision","detecte"])

print(f">>> Enregistrement '{condition}'. Augmente la lumiere TRES progressivement.")
print(">>> Reste immobile. Ctrl+C pour arreter.\n")
print(f"{'heure':8} | {'lum':>5} | {'detect':>6} | {'identite':12} | {'dist':>6}")
print("-"*52)
n=0; nd=0
try:
    while True:
        bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        lum = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
        t0 = time.perf_counter(); faces = app.get(bgr); lat = (time.perf_counter()-t0)*1000
        h = time.strftime("%H:%M:%S"); n += 1
        if not faces:
            print(f"{h} | {lum:5.1f} | {'NON':>6} | {'-':12} | {'-':>6}")
            w.writerow([h, condition, "AUCUN", "", f"{lum:.1f}", f"{lat:.0f}", "NON_DETECTE", 0])
        else:
            nd += 1
            fc = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            e = np.array(fc.normed_embedding, float)
            dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
            ok = dist <= SEUIL; nom = NOMS[i] if ok else "INCONNU"
            print(f"{h} | {lum:5.1f} | {'OUI':>6} | {nom:12} | {dist:6.3f}")
            w.writerow([h, condition, nom, f"{dist:.4f}", f"{lum:.1f}", f"{lat:.0f}", "ACCEPTE" if ok else "REFUSE", 1])
        f.flush(); time.sleep(0.4)
except KeyboardInterrupt:
    print(f"\n--- {n} images | detection {nd/max(n,1)*100:.1f}% ---")
finally:
    f.close()
    try: cam.stop()
    except Exception: pass
