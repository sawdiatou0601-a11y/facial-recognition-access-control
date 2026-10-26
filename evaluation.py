import pickle, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)
X = np.array(data["embeddings"], dtype=np.float64)
noms = np.array(data["noms"])

X = X / np.linalg.norm(X, axis=1, keepdims=True)
dist = 1.0 - (X @ X.T)

n = len(noms)
iu = np.triu_indices(n, k=1)
d = dist[iu]
same = noms[iu[0]] == noms[iu[1]]
genuine, impostor = d[same], d[~same]

print(f"Paires intra (meme personne)   : {len(genuine)}")
print(f"Paires inter (personnes diff.) : {len(impostor)}")
print(f"Intra : moyenne={genuine.mean():.4f}  ecart-type={genuine.std():.4f}")
print(f"Inter : moyenne={impostor.mean():.4f}  ecart-type={impostor.std():.4f}")

ts = np.linspace(0, 2, 2001)
FAR = np.array([(impostor <= t).mean() for t in ts])
FRR = np.array([(genuine > t).mean() for t in ts])
TAR = 1 - FRR
idx = int(np.argmin(np.abs(FAR - FRR)))
eer, t_eer = (FAR[idx] + FRR[idx]) / 2, ts[idx]
auc = float(np.trapz(TAR, FAR))

Ng, Ni = len(genuine), len(impostor)
def ligne(t):
    far = (impostor <= t).mean(); frr = (genuine > t).mean()
    acc = 1 - (far * Ni + frr * Ng) / (Ng + Ni)
    print(f"seuil={t:.3f}  FAR={far*100:5.2f}%  FRR={frr*100:5.2f}%  ACC={acc*100:5.2f}%")
for t in (0.30, 0.40, t_eer):
    ligne(t)
print(f"EER = {eer*100:.2f}%  (seuil {t_eer:.3f})    AUC = {auc:.4f}")

plt.figure(figsize=(7, 4))
plt.hist(genuine, bins=40, alpha=0.6, density=True, label="Intra (meme personne)")
plt.hist(impostor, bins=40, alpha=0.6, density=True, label="Inter (personnes diff.)")
plt.axvline(0.30, color="k", ls="--", label="Seuil 0.30")
plt.xlabel("Distance cosinus"); plt.ylabel("Densite")
plt.title("Distributions des distances intra / inter"); plt.legend()
plt.tight_layout(); plt.savefig("logs/distributions.png", dpi=150)

plt.figure(figsize=(5, 5))
plt.plot(FAR, TAR); plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.scatter([FAR[idx]], [TAR[idx]], color="red", zorder=5, label=f"EER={eer*100:.1f}%")
plt.xlabel("FAR"); plt.ylabel("TAR"); plt.title(f"Courbe ROC (AUC={auc:.3f})")
plt.legend(); plt.tight_layout(); plt.savefig("logs/roc.png", dpi=150)

with open("logs/far_frr.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seuil", "FAR", "FRR"])
    for t in np.linspace(0, 1, 101):
        w.writerow([f"{t:.3f}", f"{(impostor<=t).mean():.4f}", f"{(genuine>t).mean():.4f}"])
print("Figures: logs/distributions.png, logs/roc.png | Table: logs/far_frr.csv")
