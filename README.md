# ✍️ InvisibleInk — Write in the Air with Your Finger

> A real-time computer vision app that turns your index finger into a pen. Point to draw, open your palm to stop. That's it.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

InvisibleInk tracks your hand through a webcam and lets you **write words, draw shapes, or sketch anything** — just by moving your index finger in the air.

| Gesture | Action |
|---------|--------|
| ☝️ **Point** (index finger up, others down) | Draw on screen |
| 🖐️ **Open palm** (all fingers up) | Pen up — stop drawing |
| ✊ **Fist** | Pen up — stop drawing |
| ✌️ **Peace sign** | Idle (reserved for future eraser) |

---

## 🛠️ Tech Stack

| Component | Role |
|-----------|------|
| **MediaPipe Hands** | Tracks 21 hand keypoints at 30+ FPS via the Tasks API (WASM-backed) |
| **OpenCV** | Camera capture, rendering, and screenshot export |
| **NumPy** | Frame manipulation and overlay compositing |
| **Gesture Engine** | Custom finger-state classifier (tip vs knuckle Y-position) |

---

## 🚀 Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/YatindraRai002/InvisibleInk.git
cd InvisibleInk
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python air_writer.py
```

> On first run, the hand tracking model (~8 MB) is **auto-downloaded** from Google's servers and cached in `models/`.

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `C` | Cycle through 7 ink colors |
| `X` | Clear entire canvas |
| `Z` | Undo last stroke |
| `+` / `-` | Increase / decrease line thickness |
| `S` | Save screenshot as PNG |
| `Q` / `ESC` | Quit |

---

## 🎨 Available Colors

`Cyan` → `Magenta` → `Green` → `Orange` → `White` → `Yellow` → `Red`

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
┌─────────────────────────┐
│  MediaPipe HandLandmarker│  ← Detects 21 keypoints per hand
│  (runs locally via WASM) │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Gesture Classifier     │  ← Checks which fingers are extended
│   (tip.y vs knuckle.y)   │     to determine POINT / OPEN / FIST
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   AirCanvas Engine       │  ← Manages strokes, smoothing, undo
│   - begin_stroke()       │
│   - add_point() + smooth │
│   - end_stroke()         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Render Compositor      │  ← Draws on webcam frame:
│   - Webcam background    │     glow layer + core line + cursor
│   - Stroke overlays      │
│   - HUD                  │
└─────────────────────────┘
```

### Gesture Detection Logic

Each finger is classified as **extended** or **curled** by comparing the Y-coordinate of its tip landmark vs its PIP (middle knuckle) joint:

- **Index finger**: tip `[8]` vs PIP `[6]`
- **Middle finger**: tip `[12]` vs PIP `[10]`
- **Ring finger**: tip `[16]` vs PIP `[14]`
- **Pinky**: tip `[20]` vs PIP `[18]`
- **Thumb**: compared via X-distance from wrist `[0]`

If only the index finger is extended → `POINT` → **draw mode**.

### Line Smoothing

Raw fingertip coordinates jitter frame-to-frame. InvisibleInk applies **exponential moving average** smoothing:

```
smoothed_x = prev_x × 0.45 + raw_x × 0.55
```

Points closer than 4px apart are skipped to avoid blobs.

---

## 📁 Project Structure

```
InvisibleInk/
├── air_writer.py              # Main application (single file)
├── requirements.txt           # Python dependencies
├── models/
│   └── hand_landmarker.task   # Auto-downloaded on first run
├── README.md
└── .venv/                     # Virtual environment (not committed)
```

---

## 📦 Requirements

```
opencv-python >= 4.8.0
mediapipe >= 0.10.0
numpy >= 1.24.0
```

- **Python**: 3.10+
- **OS**: Windows, macOS, Linux
- **Camera**: Any USB webcam or built-in laptop camera

---

## 🔧 Configuration

Edit the top of `air_writer.py` to tweak:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Camera device index |
| `FRAME_W, FRAME_H` | `1280, 720` | Requested camera resolution |
| `SMOOTHING` | `0.45` | Line smoothing factor (0 = none, 1 = max) |
| `MIN_DRAW_DIST` | `4` | Minimum pixel distance between points |
| `DEFAULT_THICKNESS` | `4` | Default ink line thickness |

---

## 🗺️ Roadmap

- [ ] Eraser mode (peace sign gesture)
- [ ] Text recognition (integrate with Tesseract OCR)
- [ ] Multi-hand drawing (two people writing simultaneously)
- [ ] Export drawing only (transparent background PNG)
- [ ] Web version (browser-based with MediaPipe WASM)

---

## 📄 License

MIT — do whatever you want with it.

---

**Built with 🖐️ + 📸 + ☕ by [YatindraRai002](https://github.com/YatindraRai002)**
