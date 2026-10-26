import pickle, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pickle.load(open("encodings/buffalo.pkl","rb"))
X = np.array(d["embeddings"], float); noms = np.array(d["noms"])
X = X / np.linalg.norm(X, axis=1, keepdims=True)
D = 1 - X @ X.T; np.fill_diagonal(D, np.inf); n = len(noms)

# Identification 1:N (leave-one-out)
nn = np.argmin(D, axis=1); rank1 = float(np.mean(noms[nn] == noms))

# Distributions genuine / impostor (plus proche voisin)
g, i = [], []
for k in range(n):
    s = (noms == noms[k]).copy(); s[k] = False; f = noms != noms[k]
    if s.any(): g.append(D[k, s].min())
    if f.any(): i.append(D[k, f].min())
g, i = np.array(g), np.array(i); Ng, Ni = len(g), len(i)

ts = np.linspace(0, 2, 2001)
FAR = np.array([(i <= t).mean() for t in ts])
FRR = np.array([(g > t).mean() for t in ts])
TAR = 1 - FRR
idx = int(np.argmin(np.abs(FAR - FRR)))
eer, t_eer = (FAR[idx] + FRR[idx]) / 2, ts[idx]
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
auc = float(trap(TAR, FAR))

print(f"\n=== METRIQUES buffalo_l (galerie {n} empreintes, {len(set(noms))} personnes) ===")
print(f"Taux de reconnaissance rang-1 : {rank1*100:.2f}%")
print(f"Genuine  : moyenne={g.mean():.4f}  ecart-type={g.std():.4f}  max={g.max():.4f}")
print(f"Impostor : moyenne={i.mean():.4f}  ecart-type={i.std():.4f}  min={i.min():.4f}")
print(f"\n{'Seuil':>6} | {'FAR %':>7} | {'FRR %':>7} | {'Exactitude %':>12}")
for t in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, t_eer]:
    far = (i <= t).mean()*100; frr = (g > t).mean()*100
    acc = (1 - ((g > t).sum() + (i <= t).sum())/(Ng+Ni))*100
    print(f"{t:6.3f} | {far:7.2f} | {frr:7.2f} | {acc:12.2f}")
print(f"\nEER = {eer*100:.2f}%  (seuil {t_eer:.3f})     AUC = {auc:.4f}")
print(f"Marge de securite : genuine max = {g.max():.3f} | impostor min = {i.min():.3f}")

plt.figure(figsize=(7,4))
plt.hist(g, bins=30, alpha=0.6, density=True, label="Genuine (meme personne)")
plt.hist(i, bins=30, alpha=0.6, density=True, label="Impostor (autre personne)")
plt.axvline(0.50, color="k", ls="--", label="Seuil 0.50")
plt.xlabel("Distance cosinus au plus proche voisin"); plt.ylabel("Densite")
plt.title("buffalo_l - distributions genuine / impostor"); plt.legend()
plt.tight_layout(); plt.savefig("logs/buffalo_distributions.png", dpi=150)

plt.figure(figsize=(5,5))
plt.plot(FAR, TAR); plt.plot([0,1],[0,1],"k--",alpha=0.4)
plt.scatter([FAR[idx]],[TAR[idx]], color="red", zorder=5, label=f"EER={eer*100:.2f}%")
plt.xlabel("FAR"); plt.ylabel("TAR"); plt.title(f"ROC buffalo_l (AUC={auc:.4f})")
plt.legend(); plt.tight_layout(); plt.savefig("logs/buffalo_roc.png", dpi=150)

with open("logs/buffalo_far_frr.csv","w",newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(["seuil","FAR","FRR"])
    for t in np.linspace(0,1,101):
        w.writerow([f"{t:.3f}", f"{(i<=t).mean():.4f}", f"{(g>t).mean():.4f}"])
print("\nFigures: logs/buffalo_distributions.png, logs/buffalo_roc.png | Table: logs/buffalo_far_frr.csv")
