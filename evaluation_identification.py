import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)
X = np.array(data["embeddings"], dtype=np.float64)
noms = np.array(data["noms"])
X = X / np.linalg.norm(X, axis=1, keepdims=True)
D = 1.0 - (X @ X.T)
np.fill_diagonal(D, np.inf)          # on ne compare pas une image a elle-meme
n = len(noms)

# Taux de reconnaissance rang-1 (leave-one-out)
nn = np.argmin(D, axis=1)
rank1 = float(np.mean(noms[nn] == noms))

# Meilleure distance vers la MEME personne (genuine) et vers une AUTRE (impostor)
g_best, i_best = [], []
for k in range(n):
    same = (noms == noms[k]).copy(); same[k] = False
    diff = noms != noms[k]
    if same.any():
        g_best.append(D[k, same].min())
    if diff.any():
        i_best.append(D[k, diff].min())
g_best, i_best = np.array(g_best), np.array(i_best)

print(f"Images                          : {n}")
print(f"Taux de reconnaissance rang-1   : {rank1*100:.2f}%")
print(f"Genuine  (plus proche, meme)    : moyenne={g_best.mean():.4f}  ecart-type={g_best.std():.4f}")
print(f"Impostor (plus proche, autre)   : moyenne={i_best.mean():.4f}  ecart-type={i_best.std():.4f}")

ts = np.linspace(0, 2, 2001)
FAR = np.array([(i_best <= t).mean() for t in ts])
FRR = np.array([(g_best > t).mean() for t in ts])
TAR = 1 - FRR
idx = int(np.argmin(np.abs(FAR - FRR)))
eer, t_eer = (FAR[idx] + FRR[idx]) / 2, ts[idx]
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
auc = float(trap(TAR, FAR))

for t in (0.30, t_eer):
    print(f"seuil={t:.3f}  FAR={(i_best<=t).mean()*100:5.2f}%  FRR={(g_best>t).mean()*100:5.2f}%")
print(f"EER = {eer*100:.2f}%  (seuil {t_eer:.3f})    AUC = {auc:.4f}")

plt.figure(figsize=(7, 4))
plt.hist(g_best, bins=30, alpha=0.6, density=True, label="Genuine (meme personne)")
plt.hist(i_best, bins=30, alpha=0.6, density=True, label="Impostor (autre personne)")
plt.axvline(0.30, color="k", ls="--", label="Seuil 0.30")
plt.xlabel("Distance au plus proche voisin"); plt.ylabel("Densite")
plt.title("Identification 1:N - distributions des meilleurs matchs"); plt.legend()
plt.tight_layout(); plt.savefig("logs/identification_distributions.png", dpi=150)

plt.figure(figsize=(5, 5))
plt.plot(FAR, TAR); plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.scatter([FAR[idx]], [TAR[idx]], color="red", zorder=5, label=f"EER={eer*100:.1f}%")
plt.xlabel("FAR"); plt.ylabel("TAR"); plt.title(f"ROC identification (AUC={auc:.3f})")
plt.legend(); plt.tight_layout(); plt.savefig("logs/identification_roc.png", dpi=150)
print("Figures: logs/identification_distributions.png, logs/identification_roc.png")
