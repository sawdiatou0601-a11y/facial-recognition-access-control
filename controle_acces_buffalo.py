import sys, pickle, time, io, socketserver, threading
from http import server
from threading import Condition, Lock
import numpy as np, cv2
from picamera2 import Picamera2
from insightface.app import FaceAnalysis
import db_audit
from controle_peripheriques import Peripheriques
from ecran_lcd import EcranLCD

SEUIL = 0.50
COOLDOWN = 15
latest = {"bgr": None}; annot = {"box": None, "label": "", "color": (0,255,0)}; lock = Lock()

class SO(io.BufferedIOBase):
    def __init__(self): self.frame=None; self.condition=Condition()
    def write(self, buf):
        with self.condition: self.frame=buf; self.condition.notify_all()
stream = SO()
PAGE = "<html><body style='margin:0;background:#111;text-align:center'><img src='stream.mjpg' width='640'></body></html>"
class H(server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=='/':
            self.send_response(301); self.send_header('Location','/index.html'); self.end_headers()
        elif self.path=='/index.html':
            c=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html'); self.send_header('Content-Length',len(c)); self.end_headers(); self.wfile.write(c)
        elif self.path=='/stream.mjpg':
            self.send_response(200); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=FRAME'); self.end_headers()
            try:
                while True:
                    with stream.condition:
                        stream.condition.wait(); fr=stream.frame
                    self.wfile.write(b'--FRAME\r\n'); self.send_header('Content-Type','image/jpeg'); self.send_header('Content-Length',len(fr)); self.end_headers(); self.wfile.write(fr); self.wfile.write(b'\r\n')
            except Exception: pass
        else:
            self.send_error(404); self.end_headers()
class SS(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address=True; daemon_threads=True

def flux(cam):
    while True:
        bgr = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        with lock:
            latest["bgr"]=bgr.copy(); box=annot["box"]; lab=annot["label"]; col=annot["color"]
        if box is not None:
            x1,y1,x2,y2 = box; cv2.rectangle(bgr,(x1,y1),(x2,y2),col,2)
            if lab: cv2.putText(bgr,lab,(x1,max(15,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        ok,jpg = cv2.imencode('.jpg',bgr)
        if ok: stream.write(jpg.tobytes())
        time.sleep(0.03)

def get_frame():
    with lock: return None if latest["bgr"] is None else latest["bgr"].copy()

def main():
    sim = "--sim" in sys.argv
    db_audit.init_db()
    d = pickle.load(open("encodings/buffalo.pkl","rb"))
    G = np.array(d["embeddings"], float); N = np.array(d["noms"])
    print(f"{len(N)} empreintes buffalo_l chargees.")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
    periph = Peripheriques(simulation=sim); ecran = EcranLCD(simulation=sim)
    cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(640,480),"format":"RGB888"})); cam.start(); time.sleep(2)
    threading.Thread(target=flux, args=(cam,), daemon=True).start()
    threading.Thread(target=lambda: SS(('',8000),H).serve_forever(), daemon=True).start()
    print(">>> Apercu : tunnel SSH puis http://localhost:8000")
    dernier={}; ecran.message("Systeme pret","Approchez-vous"); print(f"Surveillance active (seuil {SEUIL}). Ctrl+C pour arreter.")
    try:
        while True:
            bgr = get_frame()
            if bgr is None: time.sleep(0.2); continue
            faces = app.get(bgr)
            if not faces:
                with lock: annot["box"], annot["label"] = None, ""
                time.sleep(0.2); continue
            f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            e = np.array(f.normed_embedding, float)
            dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
            autorise = dist <= SEUIL
            personne = N[i] if autorise else "INCONNU"
            b = f.bbox.astype(int)
            with lock:
                annot["box"] = (b[0],b[1],b[2],b[3])
                annot["label"] = f"{personne} {dist:.2f}"
                annot["color"] = (0,255,0) if autorise else (0,0,255)
            if time.time() - dernier.get(personne,0) < COOLDOWN:
                time.sleep(0.2); continue
            dernier[personne] = time.time()
            decision = "AUTORISE" if autorise else "REFUSE"
            ecran.afficher(personne, decision); db_audit.log_acces(personne, decision, dist, seuil=SEUIL)
            print(f"{time.strftime('%H:%M:%S')} {decision:8} {personne:12} d={dist:.4f}")
            if autorise: periph.acces_autorise()
            else: periph.acces_refuse()
            ecran.message("Systeme pret","Approchez-vous")
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        periph.cleanup(); ecran.cleanup()
        try: cam.stop()
        except Exception: pass
        print("Systeme arrete.")

if __name__ == "__main__":
    main()
