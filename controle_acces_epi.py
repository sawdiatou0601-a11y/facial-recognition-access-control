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
MODEL = "Facenet512"
DETECTEUR = "retinaface"     # RetinaFace seul (pas de SSD)
SEUIL = 0.30
COOLDOWN = 15

latest = {"bgr": None}; lock = Lock()

class StreamingOutput(io.BufferedIOBase):
    def __init__(self): self.frame = None; self.condition = Condition()
    def write(self, buf):
        with self.condition: self.frame = buf; self.condition.notify_all()
stream = StreamingOutput()
PAGE = "<html><body style='margin:0;background:#111;text-align:center'><img src='stream.mjpg' width='640'></body></html>"
class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(301); self.send_header('Location', '/index.html'); self.end_headers()
        elif self.path == '/index.html':
            c = PAGE.encode(); self.send_response(200); self.send_header('Content-Type', 'text/html'); self.send_header('Content-Length', len(c)); self.end_headers(); self.wfile.write(c)
        elif self.path == '/stream.mjpg':
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME'); self.end_headers()
            try:
                while True:
                    with stream.condition:
                        stream.condition.wait(); frame = stream.frame
                    self.wfile.write(b'--FRAME\r\n'); self.send_header('Content-Type', 'image/jpeg'); self.send_header('Content-Length', len(frame)); self.end_headers(); self.wfile.write(frame); self.wfile.write(b'\r\n')
            except Exception: pass
        else:
            self.send_error(404); self.end_headers()
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True; daemon_threads = True

def flux(cam):
    while True:
        bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        with lock: latest["bgr"] = bgr.copy()
        ok, jpg = cv2.imencode('.jpg', bgr)
        if ok: stream.write(jpg.tobytes())
        time.sleep(0.03)

def get_frame():
    with lock: return None if latest["bgr"] is None else latest["bgr"].copy()
def charger():
    d = pickle.load(open(ENCODINGS_PATH, "rb")); return np.array(d["embeddings"], float), np.array(d["noms"])
def dcos(a, b): return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def identifier(emb, refs, noms):
    dd = np.array([dcos(emb, r) for r in refs]); i = int(np.argmin(dd)); return noms[i], float(dd[i])

def analyser(refs, noms):
    # Reconnaissance directe : une image, RetinaFace, decision immediate
    bgr = get_frame()
    if bgr is None: return None, None, 0
    try:
        rep = DeepFace.represent(img_path=bgr, model_name=MODEL, enforce_detection=True, detector_backend=DETECTEUR)
        emb = np.array(rep[0]["embedding"])
    except Exception:
        return None, None, 0
    nom, dist = identifier(emb, refs, noms)
    return (nom if dist <= SEUIL else None), dist, 1

def main():
    sim = "--sim" in sys.argv
    db_audit.init_db(); refs, noms = charger(); print(f"{len(noms)} empreintes chargees.")
    periph = Peripheriques(simulation=sim); ecran = EcranLCD(simulation=sim)
    cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})); cam.start(); time.sleep(2)
    threading.Thread(target=flux, args=(cam,), daemon=True).start()
    threading.Thread(target=lambda: StreamingServer(('', 8000), Handler).serve_forever(), daemon=True).start()
    print(">>> Apercu : tunnel SSH puis http://localhost:8000")
    dernier = {}; ecran.message("Systeme pret", "Approchez-vous"); print("Surveillance active. Ctrl+C pour arreter.")
    try:
        while True:
            periph.attendre_presence(timeout=None)
            ecran.message("Analyse en", "cours...")
            nom, dist, visages = analyser(refs, noms)
            if visages < 1:
                ecran.message("Systeme pret", "Approchez-vous"); continue
            autorise = nom is not None
            personne = nom if autorise else "INCONNU"
            decision = "AUTORISE" if autorise else "REFUSE"
            if time.time() - dernier.get(personne, 0) < COOLDOWN:
                ecran.message("Systeme pret", "Approchez-vous"); continue
            dernier[personne] = time.time()
            ecran.afficher(personne, decision); db_audit.log_acces(personne, decision, dist, seuil=SEUIL)
            info = f"d={dist:.4f}" if dist is not None else "-"
            print(f"{time.strftime('%H:%M:%S')} {decision:8} {personne:12} {info}")
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
