# AI-Powered Facial Recognition for Physical Access Control in the Pharmaceutical Industry

**University research prototype · Raspberry Pi 5 · On-device AI**

Master's research project, Université Cheikh Anta Diop de Dakar (UCAD), Senegal.
Author: **Sawdiatou Akapo**

> **Demo video (1 min 41 s):** [Watch the live demo](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## The problem

Pharmaceutical sites protect their GMP critical areas (production, quality control, storage) with keys or badges. Keys and badges can be lost, lent or stolen, and they do not prove *who* actually entered. Staff in these areas also wear masks and hair caps, which makes many face recognition systems fail.

## The solution

A low-cost embedded system that opens a door only for enrolled staff, recognized by their face, **even when they wear a surgical mask and a hair cap**. All the AI runs locally on a Raspberry Pi 5: no cloud service and no paid API.

| Without a mask | With a mask | Mask + hair cap | Two people |
|:---:|:---:|:---:|:---:|
| ![Step 1](step1_access_granted.jpg) | ![Step 2](step2_mask.jpg) | ![Step 3](step3_mask_hair_cap.jpg) | ![Step 5](step5_two_people.jpg) |

**Green frame = identity verified, access granted.** An unknown person gets no match and access is refused.

## How it works

```
Pi Camera ──► Face detection ──► Face embedding ──► Identity match ──► Access decision
(Picamera2,    (SCRFD)            (ArcFace,          (cosine similarity   (relay + lock,
 OpenCV)                           512-D vector)      vs enrolled staff,   LCD, buzzer)
                                                      SQLite)
```

1. **Capture**: the camera streams frames to the Raspberry Pi 5.
2. **Detection**: SCRFD locates each face in the frame.
3. **Embedding**: ArcFace converts each face into a 512-dimension vector.
4. **Matching**: the vector is compared with the enrolled staff (stored in SQLite) using cosine similarity and a decision threshold.
5. **Decision**: if the face matches, the relay releases the solenoid lock, the LCD shows the authorized name and the buzzer confirms. Otherwise access is refused.

Pretrained models: InsightFace `buffalo_l` (SCRFD + ArcFace), executed with ONNX Runtime. No training is done on the board. `buffalo_l` was selected for its better results on masked faces.

## Features demonstrated

- Facial recognition without a mask
- Facial recognition with a surgical mask
- Facial recognition with a mask **and** a hair cap (cleanroom attire)
- Detection of unknown individuals (access refused)
- Recognition of multiple people in the same frame

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 5 (4 GB RAM) | Runs the whole AI pipeline on-device |
| Pi Camera NoIR V2 (Sony IMX219, 8 MP) | Captures the face |
| LED light source | Illuminates the face |
| PIR sensor | Detects a person approaching |
| Relay module | Switches the door lock |
| 5 V solenoid lock (fail-secure) | Physical door lock |
| Adjustable AC/DC power adapter set to 5 V | Separate power supply for the lock |
| 16x2 LCD (I2C) | Displays the access status |
| Buzzer, blue and red LEDs | Audible and visual feedback |
| Breadboard and jumper wires | Circuit assembly |

![Hardware](hardware_camera.jpg)

## Software stack

- Python 3
- Picamera2, OpenCV
- InsightFace (`buffalo_l`: SCRFD detection, ArcFace recognition)
- ONNX Runtime, NumPy
- gpiozero / lgpio (GPIO control)
- RPLCD / smbus2 (LCD over I2C)
- SQLite (enrolled staff database)

## Installation (Raspberry Pi OS)

```bash
# System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-lgpio i2c-tools

# Project
git clone https://github.com/sawdiatou0601-a11y/facial-recognition-access-control.git
cd facial-recognition-access-control
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

The `buffalo_l` models are downloaded automatically by InsightFace the first time the program runs.

## Usage

```bash
# 1. Enroll an authorized person
python3 YOUR_ENROLL_SCRIPT.py

# 2. Start access control
python3 YOUR_MAIN_SCRIPT.py
```

## Privacy and data protection

Face images and embeddings are personal biometric data. They are stored **only on the device** and are **never included in this repository** (see `.gitignore`). The system is designed with GMP data integrity principles (ALCOA+) and data protection requirements in mind.

## Limitations and next steps

This is a university prototype built with low-cost components. An industrial deployment in a pharmaceutical facility will require:

- a more powerful edge AI platform (for example NVIDIA Jetson);
- industrial cameras with liveness detection (anti-spoofing against photos and screens);
- certified access control hardware;
- a secure, tamper-evident audit trail of every access attempt, aligned with GMP and ALCOA+;
- validation on a larger group of users, with measured false acceptance and false rejection rates (FAR / FRR).

## Author

**Sawdiatou Akapo**
Master 2 in Applied Physics: Electronics, Systems and Telecommunications (Networks), UCAD, Dakar
Contact: akaposawdiatou@gmail.com
