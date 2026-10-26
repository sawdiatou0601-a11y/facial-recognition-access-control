import sys, pickle, time
import numpy as np
import cv2
from picamera2 import Picamera2
from deepface import DeepFace

import db_audit
from controle_peripheriques import Peripheriques
from ecran_lcd import EcranLCD

ENCODINGS_PATH = "encodings/encodings.pkl"
MODEL_NAME = "Facenet512"
DETECTORS = ("ssd", "opencv")
SEUIL = 0.30
COOLDOWN = 15
N_FRAMES = 8          # images max analysees
VOTES_MIN = 2         # votes concordants requis pour autoriser
MIN_VISAGES = 2       # visages reels min pour considerer une presentation

def charger():
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return np.array(data["embeddings"]), np.array(data["noms"])

def dcos(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identifier(emb, refs, noms):
    d = np.array([dcos(emb, r) for r in refs])
    i = int(np.argmin(d))
    return noms[i], float(d[i])

def representer(frame_bgr):
    for backend in DETECTORS:
        try:
            rep = DeepFace.represent(img_path=frame_bgr, model_name=MODEL_NAME,
                                     enforce_detection=True, detector_backend=backend)
            return np.array(rep[0]["embedding"])
        except Exception:
            continue
    return None

def analyser(cam, refs, noms):
    """Retourne (nom, distance, votes, visages_detectes).
    Sortie anticipee des qu'un utilisateur atteint VOTES_MIN."""
    votes = {}
    visages = 0
    for _ in range(N_FRAMES):
        frame_bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        emb = representer(frame_bgr)
        if emb is None:
            continue
        visages += 1
        nom, dist = identifier(emb, refs, noms)
        if dist <= SEUIL:
            votes.setdefault(nom, []).append(dist)
            gagnant = max(votes, key=lambda k: len(votes[k]))
            if len(votes[gagnant]) >= VOTES_MIN:
                return gagnant, min(votes[gagnant]), len(votes[gagnant]), visages
        time.sleep(0.1)
    if votes:
        gagnant = max(votes, key=lambda k: (len(votes[k]), -min(votes[k])))
        return gagnant, min(votes[gagnant]), len(votes[gagnant]), visages
    return None, None, 0, visages

def main():
    sim = "--sim" in sys.argv
    db_audit.init_db()
    refs, noms = charger()
    print(f"{len(noms)} empreintes chargees.")
    periph = Peripheriques(simulation=sim)
    ecran = EcranLCD(simulation=sim)

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    cam.start(); time.sleep(2)

    dernier = {}
    ecran.message("Systeme pret", "Approchez-vous")
    print("Surveillance active. Ctrl+C pour arreter.")
    try:
        while True:
            periph.attendre_presence(timeout=None)
            ecran.message("Analyse en", "cours...")
            nom, dist, votes, visages = analyser(cam, refs, noms)

            if visages < MIN_VISAGES:
                # Aucun visage reel -> on ne journalise rien
                ecran.message("Systeme pret", "Approchez-vous")
                continue

            autorise = nom is not None and votes >= VOTES_MIN and votes * 2 > visages
            personne = nom if autorise else "INCONNU"
            decision = "AUTORISE" if autorise else "REFUSE"

            if time.time() - dernier.get(personne, 0) < COOLDOWN:
                ecran.message("Systeme pret", "Approchez-vous")
                continue
            dernier[personne] = time.time()

            ecran.afficher(personne, decision)
            db_audit.log_acces(personne, decision, dist, seuil=SEUIL)
            info = f"d={dist:.4f} votes={votes}" if dist is not None else "sans match"
            print(f"{time.strftime('%H:%M:%S')} {decision:8} {personne:12} {info} (visages={visages})")
            if autorise:
                periph.acces_autorise()
            else:
                periph.acces_refuse()
            ecran.message("Systeme pret", "Approchez-vous")
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        periph.cleanup(); ecran.cleanup()
        try: cam.stop()
        except Exception: pass
        print("Systeme arrete proprement.")

if __name__ == "__main__":
    main()
