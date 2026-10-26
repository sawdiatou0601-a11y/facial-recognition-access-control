import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

lignes = list(csv.DictReader(open("logs/mesures_live.csv")))
if not lignes: raise SystemExit("Aucune mesure.")
SEUIL = 0.50

att = np.array([l["attendu"] for l in lignes])
cond = np.array([l["condition"] for l in lignes])
pred = np.array([l["predit"] for l in lignes])
dist = np.array([float(l["distance"]) for l in lignes])
acc  = np.array([l["decision"] == "ACCEPTE" for l in lignes])

genuine = (att != "INCONNU") & (att != "REFUS_ATTENDU")   # sujets autorises uniquement
impostor = att == "INCONNU"         # non enrole

print(f"=== METRIQUES OPERATIONNELLES (en ligne) — {len(lignes)} presentations, seuil {SEUIL} ===\n")
print(f"{'Condition':16} | {'n':>3} | {'accept.%':>8} | {'ident.correcte%':>15} | {'dist.moy':>8}")
for c in sorted(set(cond)):
    m = cond == c
    ident = np.mean(pred[m] == att[m]) * 100
    print(f"{c:16} | {m.sum():3d} | {np.mean(acc[m])*100:8.1f} | {ident:15.1f} | {dist[m].mean():8.3f}")

FRR = np.mean(~acc[genuine]) * 100 if genuine.any() else float('nan')
FAR = np.mean(acc[impostor]) * 100 if impostor.any() else float('nan')
IDR = np.mean(pred[genuine] == att[genuine]) * 100 if genuine.any() else float('nan')
print(f"\nFRR (autorises refuses)      : {FRR:.2f} %   (n={genuine.sum()})")
print(f"FAR (inconnus acceptes)      : {FAR:.2f} %   (n={impostor.sum()})")
print(f"Identification correcte      : {IDR:.2f} %")
m = att == "REFUS_ATTENDU"
if m.any():
    print(f"Dissimulation (refus obtenu) : {np.mean(~acc[m])*100:.2f} %   (n={m.sum()})")

# EER operationnel
if genuine.any() and impostor.any():
    g, i = dist[genuine], dist[impostor]
    ts = np.linspace(0, 1.5, 1501)
    far = np.array([(i <= t).mean() for t in ts]); frr = np.array([(g > t).mean() for t in ts])
    k = int(np.argmin(np.abs(far - frr))); eer = (far[k] + frr[k]) / 2
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trap(1 - frr, far))
    print(f"EER operationnel             : {eer*100:.2f} %  (seuil {ts[k]:.3f})")
    print(f"AUC operationnel             : {auc:.4f}")

    plt.figure(figsize=(7,4))
    plt.hist(g, bins=25, alpha=.6, density=True, label="Autorises (live)")
    plt.hist(i, bins=25, alpha=.6, density=True, label="Inconnus (live)")
    plt.axvline(SEUIL, color="k", ls="--", label=f"Seuil {SEUIL}")
    plt.xlabel("Distance cosinus"); plt.ylabel("Densite")
    plt.title("Distributions en conditions reelles"); plt.legend()
    plt.tight_layout(); plt.savefig("logs/live_distributions.png", dpi=150)

    plt.figure(figsize=(5,5))
    plt.plot(far, 1-frr); plt.plot([0,1],[0,1],"k--",alpha=.4)
    plt.scatter([far[k]],[1-frr[k]], color="red", zorder=5, label=f"EER={eer*100:.2f}%")
    plt.xlabel("FAR"); plt.ylabel("TAR"); plt.title(f"ROC operationnelle (AUC={auc:.4f})")
    plt.legend(); plt.tight_layout(); plt.savefig("logs/live_roc.png", dpi=150)

# Distances par condition (boxplot)
cs = sorted(set(cond))
plt.figure(figsize=(8,4))
plt.boxplot([dist[cond==c] for c in cs], tick_labels=cs)
plt.axhline(SEUIL, color="r", ls="--", label=f"Seuil {SEUIL}")
plt.ylabel("Distance cosinus"); plt.title("Distances mesurees par condition (live)")
plt.xticks(rotation=20); plt.legend(); plt.tight_layout()
plt.savefig("logs/live_conditions.png", dpi=150)
print("\nFigures: logs/live_distributions.png, logs/live_roc.png, logs/live_conditions.png")
