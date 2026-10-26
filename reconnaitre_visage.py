#!/usr/bin/env python3
"""
reconnaitre_visage.py
Capture une image, extrait l'embedding FaceNet512, compare aux 90 embeddings
enrolés (encodings.pkl) et retourne la meilleure correspondance.
"""

import pickle
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from deepface import DeepFace

ENCODINGS_PATH = "encodings/encodings.pkl"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "ssd"
SEUIL_DISTANCE = 0.30  # cosine, Facenet512 -- a calibrer avec tes tests

def charger_encodings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeddings = np.array(data["embeddings"])  # (90, 512)
    noms = data["noms"]                        # 90 noms
    return embeddings, noms

def capturer_image():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # stabilisation exposition / balance des blancs
    frame_rgb = picam2.capture_array()
    picam2.stop()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def distance_cosine(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def reconnaitre(frame_bgr, embeddings_ref, noms_ref):
    try:
        resultat = DeepFace.represent(
            img_path=frame_bgr,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
        )
    except ValueError:
        return None, None, "AUCUN_VISAGE_DETECTE"

    embedding_capture = np.array(resultat[0]["embedding"])
    distances = np.array([distance_cosine(embedding_capture, e) for e in embeddings_ref])
    idx_min = int(np.argmin(distances))
    distance_min = float(distances[idx_min])
    nom_candidat = noms_ref[idx_min]

    if distance_min <= SEUIL_DISTANCE:
        return nom_candidat, distance_min, "ACCES_AUTORISE"
    return nom_candidat, distance_min, "ACCES_REFUSE"

def main():
    print("Chargement des encodages de reference...")
    embeddings_ref, noms_ref = charger_encodings(ENCODINGS_PATH)
    print(f"{len(noms_ref)} embeddings charges.")

    print("Capture de l'image...")
    frame = capturer_image()
    cv2.imwrite("derniere_capture.jpg", frame)

    print("Reconnaissance en cours...")
    nom, distance, statut = reconnaitre(frame, embeddings_ref, noms_ref)

    print("-" * 40)
    print(f"Statut   : {statut}")
    if nom:
        print(f"Personne : {nom}")
        print(f"Distance : {distance:.4f} (seuil = {SEUIL_DISTANCE})")
    print("-" * 40)

if __name__ == "__main__":
    main()
