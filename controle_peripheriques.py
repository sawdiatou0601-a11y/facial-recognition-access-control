import time

PIN_RELAIS = 17
PIN_LED_VERTE = 27
PIN_LED_ROUGE = 22
PIN_BUZZER = 23
PIN_PIR = 24
DUREE_OUVERTURE = 2

try:
    from gpiozero import OutputDevice, LED, Buzzer, MotionSensor
    _GPIO = True
except Exception as e:
    _GPIO = False
    _ERR = e

class Peripheriques:
    def __init__(self, simulation=False):
        self.simulation = simulation or not _GPIO
        if self.simulation:
            raison = "" if _GPIO else f" ({_ERR})"
            print(f"[PERIPH] Mode SIMULATION{raison}")
            self.relais = self.verte = self.rouge = self.buzzer = self.pir = None
            return
        self.relais = OutputDevice(PIN_RELAIS, active_high=True, initial_value=False)
        self.verte = LED(PIN_LED_VERTE)
        self.rouge = LED(PIN_LED_ROUGE)
        self.buzzer = Buzzer(PIN_BUZZER)
        self.pir = MotionSensor(PIN_PIR)
        print("[PERIPH] GPIO initialise (serrure verrouillee).")

    def attendre_presence(self, timeout=None):
        if self.simulation:
            print("[PERIPH] (sim) presence supposee")
            return True
        print("[PERIPH] Attente de presence (PIR)...")
        self.pir.wait_for_motion(timeout=timeout)
        return True

    def acces_autorise(self, duree=DUREE_OUVERTURE):
        print("[PERIPH] ACCES AUTORISE -> LED verte + buzzer + deverrouillage")
        if self.simulation:
            time.sleep(duree)
            return
        self.rouge.off()
        self.verte.on()
        self.buzzer.beep(on_time=0.1, off_time=0.1, n=2, background=True)
        self.relais.on()
        time.sleep(duree)
        self.relais.off()
        self.verte.off()

    def acces_refuse(self):
        print("[PERIPH] ACCES REFUSE -> LED rouge + buzzer long")
        if self.simulation:
            time.sleep(1)
            return
        self.verte.off()
        self.rouge.on()
        self.buzzer.on()
        time.sleep(1)
        self.buzzer.off()
        self.rouge.off()

    def cleanup(self):
        if self.simulation:
            return
        for d in (self.relais, self.verte, self.rouge, self.buzzer):
            try:
                d.off()
            except Exception:
                pass

if __name__ == "__main__":
    import sys
    sim = "--sim" in sys.argv
    p = Peripheriques(simulation=sim)
    try:
        print(">>> Test acces autorise")
        p.acces_autorise(duree=2)
        time.sleep(1)
        print(">>> Test acces refuse")
        p.acces_refuse()
    finally:
        p.cleanup()
    print("Test termine.")
