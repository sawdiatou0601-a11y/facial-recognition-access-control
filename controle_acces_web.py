import sys, pickle, time, io, socketserver, threading
from http import server
from threading import Condition, Lock
import numpy as np, cv2
from picamera2 import Picamera2
from deepface import DeepFace
import db_audit
from controle_peripheriques import Peripheriques
from ecran_lcd import EcranLCD

ENCODINGS_PATH = "encodings/encodings.pkl"
MODEL_NAME = "Facenet512"; DETECTORS = ("ssd", "opencv")
SEUIL = 0.30; COOLDOWN = 15; N_FRAMES = 8; VOTES_MIN = 2; MIN_VISAGES = 2
ANTISPOOF = False

latest = {"bgr": None}
annot = {"box": None, "label": "", "color": (0, 255, 0)}
lock = Lock()

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None; self.condition = Condition()
    def write(self, buf):
        with self.condition:
            self.frame = buf; self.condition.notify_all()
stream = StreamingOutput()
PAGE = "<html><body style='margin:0;background:#111;text-align:center'><img src='stream.mjpg' width='640'></body></html>"

class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(301); self.send_header('Location', '/index.html'); self.end_headers()
        elif self.path == '/index.html':
            c = PAGE.encode(); self.send_response(200); self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(c)); self.end_headers(); self.wfile.write(c)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME'); self.end_headers()
            try:
                while True:
                    with stream.condition:
                        stream.condition.wait(); frame = stream.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg'); self.send_header('Content-Length', len(frame)); self.end_headers()
                    self.wfile.write(frame); self.wfile.write(b'\r\n')
            except Exception: pass
        else:
            self.send_error(404); self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True; daemon_threads = True

def flux_camera(cam):
    while True:
        arr = cam.capture_array()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        with lock:
            latest["bgr"] = bgr.copy()
            box = annot["box"]; label = annot["label"]; color = annot["color"]
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(bgr, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ok, jpg = cv2.imencode('.jpg', bgr)
        if ok: stream.write(jpg.tobytes())
        time.sleep(0.03)

def get_frame():
    with lock:
        return None if latest["bgr"] is None else latest["bgr"].copy()

def charger():
    d = pickle.load(open(ENCODINGS_PATH, 'rb')); return np.array(d['embeddings']), np.array(d['noms'])
def dcos(a, b): return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def identifier(emb, refs, noms):
    d = np.array([dcos(emb, r) for r in refs]); i = int(np.argmin(d)); return noms[i], float(d[i])
def representer(bgr):
    for backend in DETECTORS:
        try:
            rep = DeepFace.represent(img_path=bgr, model_name=MODEL_NAME, enforce_detection=True, detector_backend=backend)
            return np.array(rep[0]['embedding'])
        except Exception: continue
    return None

def detecter(bgr):
    """Retourne (box, is_real) ou None si aucun visage."""
    global ANTISPOOF
    try:
        if ANTISPOOF:
            faces = DeepFace.extract_faces(bgr, detector_backend="ssd", enforce_detection=True, anti_spoofing=True)
        else:
            faces = DeepFace.extract_faces(bgr, detector_backend="ssd", enforce_detection=True)
    except TypeError:
        ANTISPOOF = False
        try:
            faces = DeepFace.extract_faces(bgr, detector_backend="ssd", enforce_detection=True)
        except Exception:
            return None
    except Exception:
        return None
    f0 = faces[0]; fa = f0["facial_area"]
    box = (int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"]))
    return box, bool(f0.get("is_real", True))

def annotateur(refs, noms):
    while True:
        bgr = get_frame()
        if bgr is None: time.sleep(0.1); continue
        det = detecter(bgr)
        if det is None:
            with lock: annot["box"], annot["label"] = None, ""
            time.sleep(0.05); continue
        box, is_real = det
        if not is_real:
            with lock: annot["box"], annot["label"], annot["color"] = box, "PHOTO ?", (0, 165, 255)
            time.sleep(0.05); continue
        emb = representer(bgr)
        if emb is None:
            with lock: annot["box"], annot["label"] = None, ""
            time.sleep(0.05); continue
        nom, dist = identifier(emb, refs, noms)
        if dist <= SEUIL:
            with lock: annot["box"], annot["label"], annot["color"] = box, f"{nom} {dist:.2f}", (0, 255, 0)
        else:
            with lock: annot["box"], annot["label"], annot["color"] = box, f"Inconnu {dist:.2f}", (0, 0, 255)
        time.sleep(0.05)

def analyser(refs, noms):
    votes = {}; visages = 0; reels = 0
    for _ in range(N_FRAMES):
        bgr = get_frame()
        if bgr is None: time.sleep(0.1); continue
        det = detecter(bgr)
        if det is None: time.sleep(0.1); continue
        box, is_real = det
        visages += 1
        if not is_real:
            time.sleep(0.1); continue
        reels += 1
        emb = representer(bgr)
        if emb is None: time.sleep(0.1); continue
        nom, dist = identifier(emb, refs, noms)
        if dist <= SEUIL:
            votes.setdefault(nom, []).append(dist)
            g = max(votes, key=lambda k: len(votes[k]))
            if len(votes[g]) >= VOTES_MIN: return g, min(votes[g]), len(votes[g]), visages, reels
        time.sleep(0.1)
    if votes:
        g = max(votes, key=lambda k: (len(votes[k]), -min(votes[k])))
        return g, min(votes[g]), len(votes[g]), visages, reels
    return None, None, 0, visages, reels

def main():
    sim = "--sim" in sys.argv
    db_audit.init_db()
    refs, noms = charger()
    periph = Peripheriques(simulation=sim); ecran = EcranLCD(simulation=sim)
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    cam.start(); time.sleep(2)
    threading.Thread(target=flux_camera, args=(cam,), daemon=True).start()
    threading.Thread(target=annotateur, args=(refs, noms), daemon=True).start()
    threading.Thread(target=lambda: StreamingServer(('', 8000), Handler).serve_forever(), daemon=True).start()
    print(">>> Apercu live : tunnel SSH puis http://localhost:8000")
    dernier = {}; ecran.message("Systeme pret", "Approchez-vous")
    print("Surveillance active. Ctrl+C pour arreter.")
    try:
        while True:
            periph.attendre_presence(timeout=None)
            ecran.message("Analyse en", "cours...")
            nom, dist, votes, visages, reels = analyser(refs, noms)
            if visages < MIN_VISAGES:
                ecran.message("Systeme pret", "Approchez-vous"); continue

            if reels < MIN_VISAGES:
                personne, decision, dist = "PHOTO", "REFUSE", None
                autorise = False
            else:
                autorise = nom is not None and votes >= VOTES_MIN and votes * 2 > reels
                personne = nom if autorise else "INCONNU"
                decision = "AUTORISE" if autorise else "REFUSE"

            if time.time() - dernier.get(personne, 0) < COOLDOWN:
                ecran.message("Systeme pret", "Approchez-vous"); continue
            dernier[personne] = time.time()

            ecran.afficher(personne, decision)
            db_audit.log_acces(personne, decision, dist, seuil=SEUIL)
            info = f"d={dist:.4f} votes={votes}" if dist is not None else "vivacite=NON"
            print(f"{time.strftime('%H:%M:%S')} {decision:8} {personne:12} {info} (visages={visages}, reels={reels})")
            if autorise: periph.acces_autorise()
            else: periph.acces_refuse()
            ecran.message("Systeme pret", "Approchez-vous")
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        periph.cleanup(); ecran.cleanup()
        try: cam.stop()
        except Exception: pass
        print("Systeme arrete.")

if __name__ == "__main__":
    main()
