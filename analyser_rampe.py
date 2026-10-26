import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = [l for l in csv.DictReader(open("logs/mesures_eclairage.csv")) if l["condition"]=="rampe"]
L = [l for l in L if l["horodatage"] < "16:31:18"]      # exclure la periode de mouvement

# Colonnes decalees pour les lignes 'rampe' : vrai nom=attendu, vraie distance=predit, vraie luminance=distance
lum, det, acc, dist = [], [], [], []
for l in L:
    try: lu = float(l["distance"])
    except: continue
    d_ = l["predit"]
    lum.append(lu)
    det.append(int(l["detecte"]))
    acc.append(l["decision"]=="ACCEPTE")
    dist.append(float(d_) if d_ not in ("", None) else np.nan)
lum=np.array(lum); det=np.array(det); acc=np.array(acc); dist=np.array(dist)
print(f"Luminances : min={lum.min():.1f} max={lum.max():.1f} | {len(lum)} images\n")

bornes = [0,2,3,4,6,10,15,20,30,40,60]
print(f"{'Tranche':12} | {'n':>4} | {'detect %':>8} | {'accept %':>8} | {'dist moy':>8}")
print("-"*54)
c_, d_, a_ = [], [], []
for x,y in zip(bornes[:-1], bornes[1:]):
    m = (lum>=x)&(lum<y)
    if m.sum()==0: continue
    md = m & (det==1)
    td = det[m].mean()*100
    ta = (acc[md].mean()*100) if md.sum() else 0.0
    dm = np.nanmean(dist[md]) if md.sum() else np.nan
    print(f"{f'[{x} - {y}[':12} | {m.sum():4d} | {td:8.1f} | {ta:8.1f} | {dm:8.3f}")
    c_.append((x+y)/2); d_.append(td); a_.append(ta)

sd = next((c for c,v in zip(c_,d_) if v>=95), None)
sf = next((c for c,vd,va in zip(c_,d_,a_) if vd>=95 and va>=90), None)
print("\n--- SEUILS ---")
print(f"Seuil de detection (>=95 %)     : luminance ~ {sd}")
print(f"Seuil de reconnaissance fiable  : luminance ~ {sf}")

fig, ax = plt.subplots(figsize=(8.5,4.8))
ax.plot(c_, d_, "o-", color="#457b9d", label="Detection (%)")
ax.plot(c_, a_, "s-", color="#2a9d8f", label="Acceptation (%)")
ax.axhline(95, color="gray", ls=":", lw=1)
if sf: ax.axvline(sf, color="red", ls="--", label=f"Seuil fiable ~{sf:.0f}")
ax.set_xlabel("Luminance moyenne (0-255)"); ax.set_ylabel("Taux (%)")
ax.set_ylim(-5,108); ax.legend(loc="lower right"); ax.grid(alpha=.3)
plt.title("Mesure en rampe : seuils de luminance du systeme")
plt.tight_layout(); plt.savefig("logs/fig_rampe.png", dpi=150)
print("Figure: logs/fig_rampe.png")
