import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEUIL = 0.50
L = list(csv.DictReader(open("logs/mesures_live.csv")))
att  = np.array([l["attendu"] for l in L])
cond = np.array([l["condition"] for l in L])
dist = np.array([float(l["distance"]) for l in L])
acc  = np.array([l["decision"] == "ACCEPTE" for l in L])

ordre = ["sans_masque", "masque_epi", "dissimulation", "inconnu_reel"]
etiq  = ["Sans masque", "Masque + EPI", "Dissimulation", "Inconnu reel"]
ordre = [c for c in ordre if (cond == c).any()]
etiq  = [etiq[i] for i, c in enumerate(["sans_masque","masque_epi","dissimulation","inconnu_reel"]) if (cond==c).any()]

# --- Figure 1 : distances par condition (boxplot)
plt.figure(figsize=(8, 4.5))
plt.boxplot([dist[cond == c] for c in ordre], tick_labels=etiq)
plt.axhline(SEUIL, color="red", ls="--", label=f"Seuil de decision ({SEUIL})")
plt.ylabel("Distance cosinus"); plt.title("Distances mesurees en conditions reelles")
plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig("logs/fig_conditions.png", dpi=150)

# --- Figure 2 : taux d'acceptation / refus par scenario
taux_acc = [np.mean(acc[cond == c]) * 100 for c in ordre]
taux_ref = [100 - t for t in taux_acc]
x = np.arange(len(ordre)); w = 0.38
plt.figure(figsize=(8, 4.5))
plt.bar(x - w/2, taux_acc, w, label="Acces accorde", color="#2a9d8f")
plt.bar(x + w/2, taux_ref, w, label="Acces refuse", color="#e76f51")
for k, (a, r) in enumerate(zip(taux_acc, taux_ref)):
    plt.text(k - w/2, a + 1.5, f"{a:.1f}", ha="center", fontsize=9)
    plt.text(k + w/2, r + 1.5, f"{r:.1f}", ha="center", fontsize=9)
plt.xticks(x, etiq); plt.ylabel("Pourcentage (%)"); plt.ylim(0, 112)
plt.title("Decisions du systeme par scenario"); plt.legend(); plt.tight_layout()
plt.savefig("logs/fig_decisions.png", dpi=150)

# --- Figure 3 : distributions autorises vs inconnus
gen = (att != "INCONNU") & (att != "REFUS_ATTENDU")
imp = att == "INCONNU"
plt.figure(figsize=(8, 4.2))
plt.hist(dist[gen], bins=20, alpha=.65, density=True, label="Utilisateurs autorises", color="#2a9d8f")
if imp.any():
    plt.hist(dist[imp], bins=8, alpha=.65, density=True, label="Personne non enrolee", color="#e76f51")
plt.axvline(SEUIL, color="k", ls="--", label=f"Seuil ({SEUIL})")
plt.xlabel("Distance cosinus"); plt.ylabel("Densite")
plt.title("Separation autorises / non autorises (conditions reelles)")
plt.legend(); plt.tight_layout(); plt.savefig("logs/fig_separation.png", dpi=150)

# --- Figure 4 : FAR et FRR selon le seuil
if gen.any() and imp.any():
    g, i = dist[gen], dist[imp]
    ts = np.linspace(0, 1.2, 601)
    FAR = np.array([(i <= t).mean() for t in ts]) * 100
    FRR = np.array([(g > t).mean() for t in ts]) * 100
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(ts, FAR, label="FAR (fausse acceptation)", color="#e76f51")
    plt.plot(ts, FRR, label="FRR (faux rejet)", color="#2a9d8f")
    plt.axvline(SEUIL, color="k", ls="--", label=f"Seuil retenu ({SEUIL})")
    plt.xlabel("Seuil de decision"); plt.ylabel("Taux (%)")
    plt.title("Compromis FAR / FRR selon le seuil (donnees reelles)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("logs/fig_far_frr.png", dpi=150)

print("Figures generees :")
print("  logs/fig_conditions.png   - distances par condition")
print("  logs/fig_decisions.png    - taux acces accorde / refuse")
print("  logs/fig_separation.png   - separation autorises / inconnus")
print("  logs/fig_far_frr.png      - compromis FAR / FRR selon le seuil")
