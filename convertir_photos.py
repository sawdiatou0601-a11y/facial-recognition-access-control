import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET = os.path.expanduser("~/reconnaissance_faciale/dataset")

for nom in os.listdir(DATASET):
    if nom.startswith('.'): continue
    dossier = os.path.join(DATASET, nom)
    if not os.path.isdir(dossier): continue
    print(f"Conversion : {nom}")
    for fichier in os.listdir(dossier):
        if fichier.startswith('.'): continue
        chemin = os.path.join(dossier, fichier)
        try:
            img = Image.open(chemin)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            nouveau_nom = os.path.splitext(fichier)[0].replace(" ", "_") + ".jpg"
            nouveau_chemin = os.path.join(dossier, nouveau_nom)
            img.save(nouveau_chemin, "JPEG", quality=95)
            if fichier != nouveau_nom:
                os.remove(chemin)
            print(f"  OK {fichier}")
        except Exception as e:
            print(f"  ERREUR {fichier} -> {e}")

print("Conversion terminee.")
