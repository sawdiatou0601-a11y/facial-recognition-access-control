import time

LCD_ADRESSE = 0x27          # 0x27 ou 0x3F selon le module (verifier avec i2cdetect)
LCD_EXPANDER = "PCF8574"

try:
    from RPLCD.i2c import CharLCD
    _LCD = True
except Exception as e:
    _LCD = False
    _ERR = e

class EcranLCD:
    def __init__(self, adresse=LCD_ADRESSE, simulation=False):
        self.simulation = simulation or not _LCD
        if self.simulation:
            raison = "" if _LCD else f" ({_ERR})"
            print(f"[LCD] Mode SIMULATION{raison}")
            self.lcd = None
            return
        self.lcd = CharLCD(LCD_EXPANDER, adresse, cols=16, rows=2, auto_linebreaks=False)
        self.lcd.clear()

    def _ligne(self, texte):
        return texte[:16].ljust(16)

    def message(self, ligne1, ligne2=""):
        l1, l2 = self._ligne(ligne1), self._ligne(ligne2)
        if self.simulation:
            print("[LCD] +----------------+")
            print(f"[LCD] |{l1}|")
            print(f"[LCD] |{l2}|")
            print("[LCD] +----------------+")
            return
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0); self.lcd.write_string(l1)
        self.lcd.cursor_pos = (1, 0); self.lcd.write_string(l2)

    def afficher(self, nom, decision, heure=None):
        heure = heure or time.strftime("%H:%M")
        self.message(nom, f"{decision} {heure}")

    def cleanup(self):
        if self.simulation or self.lcd is None:
            return
        try:
            self.lcd.clear()
        except Exception:
            pass

if __name__ == "__main__":
    import sys
    sim = "--sim" in sys.argv
    e = EcranLCD(simulation=sim)
    e.message("Systeme pret", "Approchez-vous")
    time.sleep(1)
    e.afficher("Sawdiatou", "AUTORISE")
    time.sleep(1)
    e.afficher("INCONNU", "REFUSE")
    e.cleanup()
    print("Test LCD termine.")
