import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = list(csv.DictReader(open("logs/mesures_eclairage.csv")))
lum = np.array([float(l["luminance"]) for l in L])
det = np.array([int(l["detecte"]) for l in L])
acc = np.array([l["decision"] == "ACCEPTE" for l in L])

bornes = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 90]
print(f"{'Tranche luminance':20} | {'n':>4} | {'detection %':>11} | {'acceptation %':>13}")
print("-"*60)
centres, taux_det, taux_acc = [], [], []
for a, b in zip(bornes[:-1], bornes[1:]):
    m = (lum >= a) & (lum < b)
    if m.sum() == 0: continue
    td = det[m].mean()*100
    ta = (acc[m].sum()/max(det[m].sum(),1))*100     # parmi les visages detectes
    print(f"{f'[{a} - {b}[':20} | {m.sum():4d} | {td:11.1f} | {ta:13.1f}")
    centres.append((a+b)/2); taux_det.append(td); taux_acc.append(ta)

# Seuil minimal : plus petite luminance ou detection ET acceptation >= 90%
seuil = None
for c, d_, a_ in zip(centres, taux_det, taux_acc):
    if d_ >= 90 and a_ >= 90:
        seuil = c; break
print("\n--- SEUIL MINIMAL DE LUMINANCE ---")
if seuil:
    print(f"Detection et acceptation >= 90 % a partir d'une luminance d'environ {seuil:.0f}/255")
else:
    print("Aucune tranche n'atteint 90 % sur les deux criteres (donnees insuffisantes)")

# Comparaison brute des deux regimes
bas = lum < 4; haut = (lum >= 4)
print(f"\nLuminance < 4   : detection {det[bas].mean()*100:5.1f} %  | acceptation {(acc[bas].sum()/max(det[bas].sum(),1))*100:5.1f} %  (n={bas.sum()})")
print(f"Luminance >= 4  : detection {det[haut].mean()*100:5.1f} %  | acceptation {(acc[haut].sum()/max(det[haut].sum(),1))*100:5.1f} %  (n={haut.sum()})")

plt.figure(figsize=(8,4.6))
plt.plot(centres, taux_det, "o-", label="Taux de detection (%)", color="#457b9d")
plt.plot(centres, taux_acc, "s-", label="Taux d'acceptation (%)", color="#2a9d8f")
plt.axhline(90, color="gray", ls=":", label="Exigence 90 %")
if seuil: plt.axvline(seuil, color="red", ls="--", label=f"Seuil minimal ≈ {seuil:.0f}")
plt.xlabel("Luminance moyenne de l'image (0-255)"); plt.ylabel("Taux (%)")
plt.title("Determination du seuil minimal de luminance")
plt.ylim(-5, 108); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("logs/fig_seuil_luminance.png", dpi=150)
print("\nFigure: logs/fig_seuil_luminance.png")
