import time, pickle
import numpy as np, cv2
from picamera2 import Picamera2
from deepface import DeepFace

N = 15
with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)
refs = np.array(data["embeddings"]); noms = np.array(data["noms"])

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start(); time.sleep(2)

def dcos(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Chauffe du modele (non mesuree) pour eliminer le cout de chargement initial
print(">>> Chauffe du modele (non mesure)...")
for _ in range(3):
    try:
        f0 = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        DeepFace.represent(img_path=f0, model_name="Facenet512",
                           enforce_detection=False, detector_backend="ssd")
    except Exception:
        pass

print(">>> Place-toi bien face a la camera. Mesure dans 4 secondes...")
time.sleep(4)

t_cap, t_emb, t_cmp = [], [], []
faits, essais = 0, 0
while faits < N and essais < N * 8:
    essais += 1
    a = time.perf_counter()
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    b = time.perf_counter()
    try:
        rep = DeepFace.represent(img_path=frame, model_name="Facenet512",
                                 enforce_detection=True, detector_backend="ssd")
    except Exception:
        continue
    c = time.perf_counter()
    emb = np.array(rep[0]["embedding"])
    d = np.array([dcos(emb, r) for r in refs]); int(np.argmin(d))
    e = time.perf_counter()
    t_cap.append((b - a) * 1000); t_emb.append((c - b) * 1000); t_cmp.append((e - c) * 1000)
    faits += 1
    print(f"  mesure {faits}/{N}")
cam.stop()

def stat(name, arr):
    arr = np.array(arr)
    print(f"{name:38} moyenne={arr.mean():7.1f} ms   ecart-type={arr.std():6.1f} ms")

print(f"\n--- Temps de traitement en regime etabli ({faits} mesures) ---")
stat("Acquisition image", t_cap)
stat("Detection + embedding (SSD + FaceNet512)", t_emb)
stat("Comparaison a la base + decision", t_cmp)
stat("TOTAL bout-en-bout", np.array(t_cap) + np.array(t_emb) + np.array(t_cmp))
