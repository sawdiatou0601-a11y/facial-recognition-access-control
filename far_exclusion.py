import sys, os, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
exclu = sys.argv[1]                                   # personne retiree de la galerie
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 20

d = pickle.load(open("encodings/buffalo.pkl", "rb"))
E = np.array(d["embeddings"], float); N_ALL = np.array(d["noms"])
garde = N_ALL != exclu
G = E[garde]; NOMS = N_ALL[garde]
print(f">>> Galerie complete : {len(N_ALL)} empreintes")
print(f">>> Sujet exclu      : {exclu} ({(~garde).sum()} empreintes retirees)")
print(f">>> Galerie de test  : {len(NOMS)} empreintes, {len(set(NOMS))} personnes")
if (~garde).sum() == 0:
    print("ATTENTION : aucun nom correspondant, verifie l'orthographe."); sys.exit(1)

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(f"\n>>> {exclu} se presente comme personne NON ENROLEE. {N} presentations.")
print(">>> Place-toi face a la camera. Debut dans 5 s..."); time.sleep(5)

os.makedirs("logs", exist_ok=True)
p = "logs/mesures_far.csv"; neuf = not os.path.exists(p)
f = open(p, "a", newline=""); w = csv.writer(f)
if neuf: w.writerow(["horodatage","sujet_exclu","predit","distance","decision","detecte"])

k = 0; essais = 0; accepte = 0; dists = []
while k < N and essais < N*8:
    essais += 1
    bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        time.sleep(0.4); continue
    fc = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    e = np.array(fc.normed_embedding, float)
    dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
    ok = dist <= SEUIL
    predit = NOMS[i] if ok else "INCONNU"
    if ok: accepte += 1
    dists.append(dist); k += 1
    w.writerow([time.strftime("%H:%M:%S"), exclu, predit, f"{dist:.4f}",
                "ACCEPTE" if ok else "REFUSE", 1])
    marque = "  <-- FAUSSE ACCEPTATION" if ok else ""
    print(f"  {k:2d}/{N}: {predit:12} d={dist:.3f} {'ACCEPTE' if ok else 'REFUSE '}{marque}")
    time.sleep(0.6)
f.close(); cam.stop()

dists = np.array(dists)
print(f"\n=== RESULTAT — {exclu} traitee comme non enrolee ===")
print(f"Presentations         : {k}")
print(f"Fausses acceptations  : {accepte}")
print(f"FAR                   : {accepte/max(k,1)*100:.2f} %")
print(f"Distance moyenne      : {dists.mean():.3f}  (min {dists.min():.3f} / max {dists.max():.3f})")
