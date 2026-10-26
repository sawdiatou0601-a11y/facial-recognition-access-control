import os, glob
import numpy as np, cv2
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
E, N = [], []
for personne in sorted(os.listdir("dataset")):
    d = os.path.join("dataset", personne)
    if not os.path.isdir(d): continue
    for img in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        f = app.get(cv2.imread(img))
        if not f: continue
        E.append(np.array(f[0].normed_embedding, float)); N.append(personne)
    print("fait:", personne, flush=True)
E = np.array(E); N = np.array(N); n = len(N)
D = 1 - E @ E.T; np.fill_diagonal(D, np.inf)
nn = np.argmin(D, axis=1); rank1 = float(np.mean(N[nn] == N))
g, i = [], []
for k in range(n):
    s = (N == N[k]).copy(); s[k] = False; f = N != N[k]
    if s.any(): g.append(D[k, s].min())
    if f.any(): i.append(D[k, f].min())
g, i = np.array(g), np.array(i)
ts = np.linspace(0, 2, 2001)
FAR = np.array([(i <= t).mean() for t in ts]); FRR = np.array([(g > t).mean() for t in ts])
idx = int(np.argmin(np.abs(FAR - FRR)))
print(f"\nbuffalo_l | images={n} | rang-1 = {rank1*100:.1f}%")
print(f"genuine  moy={g.mean():.3f}  max={g.max():.3f}")
print(f"impostor moy={i.mean():.3f}  min={i.min():.3f}")
print(f"EER = {(FAR[idx]+FRR[idx])/2*100:.2f}%  -> SEUIL RECOMMANDE = {ts[idx]:.3f}")
