import sys, io, socketserver, threading, time, os
from http import server
from threading import Condition, Lock
import cv2
from picamera2 import Picamera2

condition = sys.argv[1] if len(sys.argv) > 1 else "capture"
dossier = os.path.join("test_occlusion", condition)
os.makedirs(dossier, exist_ok=True)

latest = {"bgr": None}
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

def flux(cam):
    while True:
        arr = cam.capture_array(); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        with lock: latest["bgr"] = bgr
        ok, jpg = cv2.imencode('.jpg', bgr)
        if ok: stream.write(jpg.tobytes())
        time.sleep(0.03)

cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start(); time.sleep(2)
threading.Thread(target=flux, args=(cam,), daemon=True).start()
threading.Thread(target=lambda: StreamingServer(('', 8000), Handler).serve_forever(), daemon=True).start()
print(">>> Ouvre le tunnel SSH puis http://localhost:8000")
print(f">>> Les photos seront enregistrees dans : {dossier}")
n = 0
try:
    while True:
        cmd = input("Entree = capturer une photo, q = quitter : ").strip().lower()
        if cmd == 'q': break
        with lock:
            bgr = None if latest["bgr"] is None else latest["bgr"].copy()
        if bgr is None:
            print("  (pas encore d'image)"); continue
        n += 1
        chemin = os.path.join(dossier, f"{condition}_{n}.jpg")
        cv2.imwrite(chemin, bgr)
        print(f"  capturee -> {chemin}")
finally:
    cam.stop()
    print(f"\n{n} photos enregistrees dans {dossier}")
