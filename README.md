# Systeme embarque de reconnaissance faciale pour controle d'acces

Prototype developpe dans le cadre d'un memoire de Master 2
en Electronique, Systemes et Telecommunications (UCAD).

Controle d'acces biometrique destine aux zones reglementees
en environnement pharmaceutique.

## Materiel

- Raspberry Pi 5 (4 Go)
- Camera NoIR V2 (capteur IMX219)
- Module relais et serrure a solenoide 5 V, fail-secure
- Capteur de presence PIR HC-SR501
- Ecran LCD 16x2 (bus I2C), diodes de signalisation, buzzer

## Logiciel

- Python 3
- InsightFace, paquet de modeles buffalo_l (SCRFD + ArcFace)
- ONNX Runtime pour l'inference embarquee
- OpenCV, Picamera2, NumPy
- GPIOZero et lgpio pour les entrees-sorties
- SQLite pour la journalisation des acces

## Usage

(a completer)

## Donnees

Les images d'enrolement, les empreintes faciales et la base de
journalisation ne sont pas publiees dans ce depot.
