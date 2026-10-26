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
COOLDOWN = 10
SCORE_MIN = 0.60
try:
    hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    hog = None
    print("[INFO] Detecteur de silhouette indisponible (OpenCV 5) -> regle silhouette desactivee")
latest = {"bgr": None}; annots = {"list": []}; lock = Lock()

def compter_personnes(bgr):
    if hog is None: return 0
    try:
        rects, _ = hog.detectMultiScale(bgr, winStride=(8,8), padding=(8,8), scale=1.05)
        return len(rects)
    except Exception:
        return 0

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
            latest["bgr"]=bgr.copy(); lst=list(annots["list"])
        for (b, lab, col) in lst:
            cv2.rectangle(bgr,(b[0],b[1]),(b[2],b[3]),col,2)
            cv2.putText(bgr,lab,(b[0],max(15,b[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
        ok,jpg = cv2.imencode('.jpg',bgr)
        if ok: stream.write(jpg.tobytes())
        time.sleep(0.03)

def get_frame():
    with lock: return None if latest["bgr"] is None else latest["bgr"].copy()

def alerte(periph, n=4):
    if periph.simulation:
        print("[PERIPH] (sim) ALERTE : LED rouge clignotante + buzzer"); time.sleep(1); return
    for _ in range(n):
        periph.rouge.on(); periph.buzzer.on(); time.sleep(0.25)
        periph.rouge.off(); periph.buzzer.off(); time.sleep(0.25)

def main():
    sim = "--sim" in sys.argv
    db_audit.init_db()
    d = pickle.load(open("encodings/buffalo.pkl","rb"))
    G = np.array(d["embeddings"], float); N = np.array(d["noms"])
    print(f"{len(N)} empreintes chargees.")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
    periph = Peripheriques(simulation=sim); ecran = EcranLCD(simulation=sim)
    cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size":(1280,960),"format":"RGB888"})); cam.start(); time.sleep(2)
    threading.Thread(target=flux, args=(cam,), daemon=True).start()
    threading.Thread(target=lambda: SS(('',8000),H).serve_forever(), daemon=True).start()
    print(">>> Apercu : tunnel SSH puis http://localhost:8000")
    ecran.message("Systeme pret","Approchez-vous")
    print(f"Anti-talonnage actif (seuil {SEUIL}). Ctrl+C pour arreter.")
    dernier = 0.0
    try:
        while True:
            bgr = get_frame()
            if bgr is None: time.sleep(0.2); continue
            faces = app.get(bgr)
            if not faces:
                with lock: annots["list"] = []
                time.sleep(0.2); continue

            boites=[]; identites=[]; tous_ok=True; visage_flou=False
            for f in faces:
                if float(getattr(f,"det_score",1.0)) < SCORE_MIN: visage_flou=True
                e = np.array(f.normed_embedding, float)
                dd = 1 - G @ e; i = int(np.argmin(dd)); dist = float(dd[i])
                ok = dist <= SEUIL
                nom = N[i] if ok else "INCONNU"
                if not ok: tous_ok=False
                identites.append((nom, dist))
                b = f.bbox.astype(int)
                boites.append(((b[0],b[1],b[2],b[3]), f"{nom} {dist:.2f}", (0,255,0) if ok else (0,0,255)))
            with lock: annots["list"] = boites

            if time.time() - dernier < COOLDOWN:
                time.sleep(0.2); continue
            dernier = time.time()

            n = len(faces); n_pers = compter_personnes(bgr); silhouette = n_pers > n
            noms_str = ",".join(x[0] for x in identites)
            print(f"\n{time.strftime('%H:%M:%S')} Visages: {n} | Personnes: {n_pers}")
            for k,(nom,dist) in enumerate(identites):
                print(f"   Visage {k+1}: {nom} (d={dist:.3f})")

            if tous_ok and not silhouette and not visage_flou:
                print(f"   -> ACCES AUTORISE : {noms_str}")
                ecran.afficher(noms_str[:16], "AUTORISE")
                db_audit.log_acces(noms_str, "AUTORISE", identites[0][1], seuil=SEUIL)
                periph.acces_autorise()
            else:
                if silhouette: motif = "SILHOUETTE_SANS_VISAGE"
                elif visage_flou: motif = "VISAGE_NON_IDENTIFIABLE"
                elif n >= 2: motif = "TALONNAGE"
                else: motif = "INCONNU"
                print(f"   -> ACCES REFUSE ({motif}) : porte fermee + alerte")
                ecran.afficher(motif[:16], "REFUSE")
                db_audit.log_acces(motif + ":" + noms_str, "REFUSE", identites[0][1], seuil=SEUIL)
                alerte(periph)
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
