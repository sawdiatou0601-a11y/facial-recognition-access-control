import sys, os, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
attendu = sys.argv[1]                      # "Sawdiatou", "Nadine" ou "INCONNU"
condition = sys.argv[2] if len(sys.argv) > 2 else "normal"   # ex: sans_masque, masque, epi
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

d = pickle.load(open("encodings/buffalo.pkl","rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(f">>> Sujet attendu : {attendu} | condition : {condition} | {N} presentations")
print(">>> Place-toi face a la camera. Debut dans 4 s..."); time.sleep(4)

fichier = "logs/mesures_live.csv"
os.makedirs("logs", exist_ok=True)
neuf = not os.path.exists(fichier)
f = open(fichier, "a", newline=""); w = csv.writer(f)
if neuf: w.writerow(["horodatage","attendu","condition","predit","distance","decision","correct"])

k = 0; essais = 0; non_detecte = 0
while k < N and essais < N*8:
    essais += 1
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        non_detecte += 1; time.sleep(0.3); continue
    fc = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = np.array(fc.normed_embedding, float)
    dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
    accepte = dist <= SEUIL
    predit = NOMS[i] if accepte else "INCONNU"
    correct = (predit == attendu)
    k += 1
    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), attendu, condition, predit, f"{dist:.4f}",
                "ACCEPTE" if accepte else "REFUSE", int(correct)])
    print(f"  {k}/{N}: {predit:12} d={dist:.3f} {'OK' if correct else 'ERREUR'}")
    time.sleep(0.6)
f.close(); cam.stop()
print(f"\nSession terminee. Non detectes : {non_detecte}. Donnees ajoutees a {fichier}")
