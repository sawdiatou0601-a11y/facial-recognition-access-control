import sys, os, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
condition = sys.argv[1]                                  # ex: fort, normal, faible, contre_jour
attendu   = sys.argv[2] if len(sys.argv) > 2 else "Sawdiatou"
N         = int(sys.argv[3]) if len(sys.argv) > 3 else 15

d = pickle.load(open("encodings/buffalo.pkl","rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(f">>> Condition d'eclairage : {condition} | sujet : {attendu} | {N} presentations")
print(">>> Place-toi face a la camera. Debut dans 4 s..."); time.sleep(4)

os.makedirs("logs", exist_ok=True)
fichier = "logs/mesures_eclairage.csv"
neuf = not os.path.exists(fichier)
f = open(fichier, "a", newline=""); w = csv.writer(f)
if neuf: w.writerow(["horodatage","condition","attendu","predit","distance","luminance","decision","detecte"])

k = 0; essais = 0; non_detecte = 0
while k < N and essais < N*8:
    essais += 1
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    lum = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())   # luminance moyenne 0-255
    faces = app.get(bgr)
    if not faces:
        non_detecte += 1
        w.writerow([time.strftime("%H:%M:%S"), condition, attendu, "AUCUN", "", f"{lum:.1f}", "NON_DETECTE", 0])
        print(f"  (aucun visage detecte, luminance={lum:.0f})")
        time.sleep(0.4); continue
    fc = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = np.array(fc.normed_embedding, float)
    dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
    ok = dist <= SEUIL
    predit = NOMS[i] if ok else "INCONNU"
    k += 1
    w.writerow([time.strftime("%H:%M:%S"), condition, attendu, predit, f"{dist:.4f}", f"{lum:.1f}",
                "ACCEPTE" if ok else "REFUSE", 1])
    print(f"  {k}/{N}: {predit:12} d={dist:.3f} luminance={lum:5.1f}")
    time.sleep(0.5)
f.close(); cam.stop()
print(f"\nTermine. Non detectes : {non_detecte}/{essais} tentatives.")
