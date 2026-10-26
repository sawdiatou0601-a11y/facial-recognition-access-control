import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = list(csv.DictReader(open("logs/mesures_eclairage.csv")))
cond = np.array([l["condition"] for l in L])
det  = np.array([int(l["detecte"]) for l in L])
lum  = np.array([float(l["luminance"]) for l in L])
dist = np.array([float(l["distance"]) if l["distance"] else np.nan for l in L])
acc  = np.array([l["decision"]=="ACCEPTE" for l in L])

print(f"{'Condition':14} | {'n':>4} | {'lum.moy':>8} | {'detect.%':>8} | {'dist.moy':>8} | {'accept.%':>8}")
for c in ["fort","normal","faible","contre_jour"]:
    m = cond==c
    if not m.any(): continue
    md = m & (det==1)
    print(f"{c:14} | {m.sum():4d} | {lum[m].mean():8.1f} | {det[m].mean()*100:8.1f} | "
          f"{np.nanmean(dist[md]):8.3f} | {acc[md].mean()*100:8.1f}")

# Figure 1 : distance vs luminance
ok = det==1
plt.figure(figsize=(8,4.5))
for c,col in zip(["normal","fort","contre_jour","faible"],["#2a9d8f","#e9c46a","#457b9d","#e76f51"]):
    m = (cond==c) & ok
    if m.any(): plt.scatter(lum[m], dist[m], label=c, color=col, alpha=.75, s=35)
plt.axhline(0.50, color="red", ls="--", label="Seuil (0.50)")
plt.xlabel("Luminance moyenne de l'image (0-255)"); plt.ylabel("Distance cosinus")
plt.title("Influence de l'eclairage sur la performance de reconnaissance")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("logs/fig_eclairage_scatter.png", dpi=150)

# Figure 2 : taux de detection et distance par condition
cs = [c for c in ["faible","normal","fort","contre_jour"] if (cond==c).any()]
x = np.arange(len(cs)); w=0.38
fig, ax1 = plt.subplots(figsize=(8,4.5))
taux = [ (det[cond==c].mean()*100) for c in cs ]
dmoy = [ np.nanmean(dist[(cond==c)&ok]) for c in cs ]
ax1.bar(x-w/2, taux, w, color="#457b9d", label="Taux de detection (%)")
ax1.set_ylabel("Taux de detection (%)", color="#457b9d"); ax1.set_ylim(0,110)
ax2 = ax1.twinx()
ax2.bar(x+w/2, dmoy, w, color="#e76f51", label="Distance moyenne")
ax2.axhline(0.50, color="red", ls="--")
ax2.set_ylabel("Distance cosinus moyenne", color="#e76f51")
ax1.set_xticks(x); ax1.set_xticklabels(cs)
plt.title("Detection et distance selon les conditions d'eclairage")
plt.tight_layout(); plt.savefig("logs/fig_eclairage_barres.png", dpi=150)
print("\nFigures: logs/fig_eclairage_scatter.png, logs/fig_eclairage_barres.png")
