import sys, os, time, pickle, csv
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis

SEUIL = 0.50
sujet   = sys.argv[1]
support = sys.argv[2]
N       = int(sys.argv[3]) if len(sys.argv) > 3 else 10

dossier = os.path.join("preuves", f"{support}_{sujet}")
os.makedirs(dossier, exist_ok=True)

d = pickle.load(open("encodings/buffalo.pkl", "rb"))
G = np.array(d["embeddings"], float); NOMS = np.array(d["noms"])
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"}))
cam.start(); time.sleep(2)
for _ in range(2): app.get(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR))

print(f">>> Support : {support} | sujet : {sujet} | {N} presentations")
print(f">>> Images enregistrees dans : {dossier}")
print(">>> Presente le support devant la camera. Debut dans 5 s..."); time.sleep(5)

f = open("logs/mesures_presentation.csv", "a", newline=""); w = csv.writer(f)
k=0; essais=0; accepte=0; dists=[]
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

    # annotation de l'image
    b = fc.bbox.astype(int)
    couleur = (0, 255, 0) if ok else (0, 0, 255)
    cv2.rectangle(bgr, (b[0], b[1]), (b[2], b[3]), couleur, 3)
    bandeau = bgr.copy()
    cv2.rectangle(bandeau, (0, 0), (bgr.shape[1], 96), (0, 0, 0), -1)
    bgr = cv2.addWeighted(bandeau, 0.65, bgr, 0.35, 0)
    cv2.putText(bgr, f"SUPPORT : {support.upper()} de {sujet}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bgr, f"{predit}   d={dist:.3f}   {'ACCES ACCORDE' if ok else 'ACCES REFUSE'}",
                (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)
    chemin = os.path.join(dossier, f"{support}_{k:02d}_{'accepte' if ok else 'refuse'}_{dist:.3f}.jpg")
    cv2.imwrite(chemin, bgr)

    w.writerow([time.strftime("%H:%M:%S"), sujet, support, predit, f"{dist:.4f}",
                "ACCEPTE" if ok else "REFUSE", 1])
    marque = "  <-- USURPATION REUSSIE" if ok else ""
    print(f"  {k:2d}/{N}: {predit:12} d={dist:.3f} {'ACCEPTE' if ok else 'REFUSE '}{marque}")
    time.sleep(0.7)
f.close(); cam.stop()

dists = np.array(dists)
print(f"\n=== {support.upper()} de {sujet} ===")
print(f"Presentations : {k}   Acceptations : {accepte} ({accepte/max(k,1)*100:.0f} %)")
print(f"Distance moyenne : {dists.mean():.3f}  (min {dists.min():.3f} / max {dists.max():.3f})")
print(f"Images : {dossier}")
