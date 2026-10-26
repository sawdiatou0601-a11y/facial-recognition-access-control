import os, pickle
import numpy as np
from deepface import DeepFace

DATASET   = os.path.expanduser("~/reconnaissance_faciale/dataset")
ENCODINGS = os.path.expanduser("~/reconnaissance_faciale/encodings/encodings.pkl")

embeddings_connus = []
noms_connus       = []

print("Demarrage de l'encodage FaceNet512...")

for nom in sorted(os.listdir(DATASET)):
    if nom.startswith('.'): continue
    dossier = os.path.join(DATASET, nom)
    if not os.path.isdir(dossier): continue
    print(f"\nTraitement : {nom}")
    for fichier in sorted(os.listdir(dossier)):
        if fichier.startswith('.'): continue
        if not fichier.lower().endswith(('.jpg','.jpeg','.png')): continue
        chemin = os.path.join(dossier, fichier)
        try:
            result = DeepFace.represent(
                img_path=chemin,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend="opencv"
            )
            embedding = result[0]["embedding"]
            embeddings_connus.append(embedding)
            noms_connus.append(nom)
            print(f"  OK {fichier}")
        except Exception as e:
            print(f"  ERREUR {fichier} -> visage non detecte")

data = {"embeddings": embeddings_connus, "noms": noms_connus}
with open(ENCODINGS, "wb") as f:
    pickle.dump(data, f)

print(f"\nEncodage termine.")
print(f"{len(embeddings_connus)} visages encodes pour {len(set(noms_connus))} personnes.")
