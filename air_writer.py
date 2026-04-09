

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import math
import os
import urllib.request


CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# Colors (BGR)
COLORS = [
    ("Cyan",    (255, 255, 0)),
    ("Magenta", (255, 0, 255)),
    ("Green",   (0, 255, 100)),
    ("Orange",  (0, 165, 255)),
    ("White",   (255, 255, 255)),
    ("Yellow",  (0, 255, 255)),
    ("Red",     (0, 0, 255)),
]

SMOOTHING = 0.45          # point smoothing (0=none, 1=max)
MIN_DRAW_DIST = 4         # min px between points to avoid blobs
DEFAULT_THICKNESS = 4
ERASER_RADIUS = 30        # eraser circle radius in px


def ensure_model(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[AIR WRITER] Downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[AIR WRITER] Saved to {path}")


def is_finger_extended(landmarks, tip_id, pip_id):
    """Check if a finger is extended (tip is above pip in screen coords)."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def detect_gesture(landmarks):
    """
    Detect hand gesture from landmarks.
    Returns: 'POINT' (index only up), 'OPEN' (all up), 'FIST' (all down), 'OTHER'
    """
    # Index: tip=8, pip=6
    # Middle: tip=12, pip=10
    # Ring: tip=16, pip=14
    # Pinky: tip=20, pip=18
    index_up = is_finger_extended(landmarks, 8, 6)
    middle_up = is_finger_extended(landmarks, 12, 10)
    ring_up = is_finger_extended(landmarks, 16, 14)
    pinky_up = is_finger_extended(landmarks, 20, 18)

    # Thumb: check x distance from wrist (works for both hands)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_up = abs(thumb_tip.x - landmarks[0].x) > abs(thumb_ip.x - landmarks[0].x)

    fingers_up = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINT"
    elif fingers_up >= 4:
        return "OPEN"
    elif fingers_up <= 1 and not index_up:
        return "FIST"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"  # two fingers = eraser could be added
    else:
        return "OTHER"



class AirCanvas:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.strokes = []        # list of strokes; each stroke = (color, thickness, [points])
        self.current_stroke = None
        self.color_idx = 0
        self.thickness = DEFAULT_THICKNESS
        self.prev_point = None   # for smoothing

    @property
    def color(self):
        return COLORS[self.color_idx][1]

    @property
    def color_name(self):
        return COLORS[self.color_idx][0]

    def begin_stroke(self, x, y):
        """Start a new stroke."""
        self.current_stroke = (self.color, self.thickness, [(x, y)])
        self.prev_point = (x, y)

    def add_point(self, x, y):
        """Add point to current stroke with smoothing."""
        if self.current_stroke is None:
            self.begin_stroke(x, y)
            return
        # Smooth
        if self.prev_point:
            sx = self.prev_point[0] * SMOOTHING + x * (1 - SMOOTHING)
            sy = self.prev_point[1] * SMOOTHING + y * (1 - SMOOTHING)
        else:
            sx, sy = x, y
        # Min distance check
        pts = self.current_stroke[2]
        if pts:
            last = pts[-1]
            if math.hypot(sx - last[0], sy - last[1]) < MIN_DRAW_DIST:
                return
        pts.append((int(sx), int(sy)))
        self.prev_point = (sx, sy)

    def end_stroke(self):
        """Finish current stroke and save it."""
        if self.current_stroke and len(self.current_stroke[2]) > 1:
            self.strokes.append(self.current_stroke)
        self.current_stroke = None
        self.prev_point = None

    def undo(self):
        if self.strokes:
            self.strokes.pop()

    def clear(self):
        self.strokes.clear()
        self.current_stroke = None
        self.prev_point = None

    def erase_at(self, x, y, radius=ERASER_RADIUS):
        """Remove any strokes that have points within radius of (x, y)."""
        surviving = []
        for stroke in self.strokes:
            color, thick, pts = stroke
            hit = False
            for px, py in pts:
                if math.hypot(px - x, py - y) < radius:
                    hit = True
                    break
            if not hit:
                surviving.append(stroke)
        self.strokes = surviving

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(COLORS)

    def draw(self, frame):
        """Render all strokes + current stroke onto frame."""
        # Draw completed strokes
        for color, thick, pts in self.strokes:
            self._draw_stroke(frame, pts, color, thick)

        # Draw current in-progress stroke
        if self.current_stroke and len(self.current_stroke[2]) > 1:
            self._draw_stroke(frame, self.current_stroke[2],
                              self.current_stroke[0], self.current_stroke[1])

    def _draw_stroke(self, frame, pts, color, thick):
        """Draw a single stroke with glow effect."""
        if len(pts) < 2:
            return
        pts_arr = np.array(pts, dtype=np.int32)

        # Glow layer
        overlay = frame.copy()
        cv2.polylines(overlay, [pts_arr], False, color, thick + 6, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Core line
        cv2.polylines(frame, [pts_arr], False, color, thick, cv2.LINE_AA)



def draw_hud(frame, canvas, gesture, fps, w, h):
    """Draw minimal HUD."""
    # Top-left info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (230, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.0f}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Color swatch
    cv2.circle(frame, (16, 52), 8, canvas.color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"Color: {canvas.color_name}", (30, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, canvas.color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Thickness: {canvas.thickness}", (16, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Gesture indicator
    gesture_colors = {
        "POINT": (0, 255, 100),
        "OPEN": (200, 200, 200),
        "FIST": (100, 100, 100),
        "PEACE": (255, 255, 0),
        "OTHER": (100, 100, 100),
        "NONE": (80, 80, 80),
    }
    gc = gesture_colors.get(gesture, (100, 100, 100))
    cv2.putText(frame, f"Gesture: {gesture}", (16, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, gc, 1, cv2.LINE_AA)

    # Drawing indicator (top-right)
    if gesture == "POINT":
        cv2.circle(frame, (w - 25, 25), 10, (0, 255, 100), -1, cv2.LINE_AA)
        cv2.putText(frame, "DRAWING", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
    elif gesture == "PEACE":
        cv2.circle(frame, (w - 25, 25), 10, (180, 100, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "ERASING", (w - 115, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 100, 255), 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (w - 25, 25), 10, (80, 80, 80), -1, cv2.LINE_AA)
        cv2.putText(frame, "PEN UP", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    # Bottom controls bar
    bar = "[C] Color  [X] Clear  [Z] Undo  [+/-] Size  [S] Save  [Q] Quit"
    tw = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    bx = (w - tw) // 2
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bx - 10, h - 28), (bx + tw + 10, h - 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, bar, (bx, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    # Strokes counter
    n = len(canvas.strokes) + (1 if canvas.current_stroke else 0)
    cv2.putText(frame, f"Strokes: {n}", (w - 120, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)


def draw_eraser_cursor(frame, x, y):
    """Draw an eraser cursor (pink dashed circle)."""
    # Outer glow
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), ERASER_RADIUS + 4, (180, 100, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    # Eraser ring
    cv2.circle(frame, (x, y), ERASER_RADIUS, (180, 100, 255), 2, cv2.LINE_AA)
    # Inner cross
    s = 6
    cv2.line(frame, (x - s, y - s), (x + s, y + s), (180, 100, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x + s, y - s), (x - s, y + s), (180, 100, 255), 1, cv2.LINE_AA)


def draw_fingertip_cursor(frame, x, y, color, drawing):
    """Draw a cursor at the fingertip position."""
    if drawing:
        # Drawing cursor: filled circle with glow
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), 14, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        # Idle cursor: hollow circle
        cv2.circle(frame, (x, y), 8, color, 1, cv2.LINE_AA)



def main():
    ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)

    # MediaPipe Hand Landmarker
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[AIR WRITER] ERROR: Could not open camera.")
        return

    ret, test = cap.read()
    if not ret:
        print("[AIR WRITER] ERROR: Could not read from camera.")
        cap.release()
        return

    h, w = test.shape[:2]
    print(f"[AIR WRITER] Camera: {w}x{h}")

    # State
    canvas = AirCanvas(w, h)
    prev_time = time.time()
    fps = 0.0
    ts_ms = 0
    was_drawing = False
    screenshot_ct = 0

    print("[AIR WRITER] Ready! Point your index finger to write in the air.")
    print("  Open palm = pen up  |  [C] color  |  [X] clear  |  [Z] undo  |  [Q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Slight darken for contrast
        tint = np.full_like(frame, (8, 5, 3), dtype=np.uint8)
        cv2.addWeighted(tint, 0.2, frame, 0.8, 0, frame)

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += 33
        results = hand_landmarker.detect_for_video(mp_image, ts_ms)

        gesture = "NONE"
        tip_x, tip_y = -1, -1

        if results.hand_landmarks:
            lms = results.hand_landmarks[0]
            gesture = detect_gesture(lms)

            # Index fingertip position
            tip = lms[8]
            tip_x = int(tip.x * w)
            tip_y = int(tip.y * h)

            is_drawing = gesture == "POINT"
            is_erasing = gesture == "PEACE"

            if is_drawing:
                if not was_drawing:
                    canvas.begin_stroke(tip_x, tip_y)
                else:
                    canvas.add_point(tip_x, tip_y)
            elif is_erasing:
                if was_drawing:
                    canvas.end_stroke()
                canvas.erase_at(tip_x, tip_y)
            else:
                if was_drawing:
                    canvas.end_stroke()

            was_drawing = is_drawing

            # Draw cursor
            if is_erasing:
                draw_eraser_cursor(frame, tip_x, tip_y)
            else:
                draw_fingertip_cursor(frame, tip_x, tip_y, canvas.color, is_drawing)
        else:
            if was_drawing:
                canvas.end_stroke()
            was_drawing = False

        # Draw all strokes
        canvas.draw(frame)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        # HUD
        draw_hud(frame, canvas, gesture, fps, w, h)

        # Display
        cv2.imshow("AIR WRITER", frame)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            canvas.cycle_color()
            print(f"[AIR WRITER] Color: {canvas.color_name}")
        elif key == ord('x'):
            canvas.clear()
            print("[AIR WRITER] Canvas cleared")
        elif key == ord('z'):
            canvas.undo()
            print("[AIR WRITER] Undo")
        elif key == ord('+') or key == ord('='):
            canvas.thickness = min(20, canvas.thickness + 1)
            print(f"[AIR WRITER] Thickness: {canvas.thickness}")
        elif key == ord('-'):
            canvas.thickness = max(1, canvas.thickness - 1)
            print(f"[AIR WRITER] Thickness: {canvas.thickness}")
        elif key == ord('s'):
            screenshot_ct += 1
            fn = f"airwriter_{screenshot_ct:03d}.png"
            fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
            cv2.imwrite(fp, frame)
            print(f"[AIR WRITER] Saved: {fp}")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    print("[AIR WRITER] Session ended.")


if __name__ == "__main__":
    main()
