#!/usr/bin/env python3

"""

demo2_pointing_gridmap_ui_demo2_polished.py

YOLO tool detection + pan/tilt + laser using:

- 7x7 manual grid calibration (laser_grid_5x5.json)

- Piecewise bilinear interpolation per grid cell

- EMA smoothing

- Deadband + rate limiting

- PWM cut to reduce jitter

- FULLSCREEN display that actually FILLS the screen (scale + center-crop)

- Clean demo UI:

   * Top status bar (MODE/LOCK/LASER/PAN-TILT/FPS + DRAWER/CABINET state)

   * Right tool panel with FIXED ORDER (no confidence-sorting, no shifting)

   * No empty slots

   * Crosshair + pulsing dot (keep it badass)

   * Toast notifications (mode changes, reed switch changes, etc.)

- Selection changes (Demo2 polish):

   * TAB cycles through tools (stable list)

   * ENTER locks selected tool

   * A sets AUTO

   * S opens search box to type tool name -> ENTER locks it

   * ESC quit

   * (m / u / [ ] / 1-9 removed)

Behavior (Demo 2):

- AUTO: target = best overall detection BUT with target-hold hysteresis to prevent bouncing

- LOCK: target = best box of locked class; if not present -> laser OFF and no servo updates (after a short grace)

- Reed switches: show OPEN/CLOSED indicators + toast on state change (NO inventory logic yet)

Classes (fixed):

- allen_keys, hammer, handsaw, pliers, screwdriver, tape_measure, wrench

"""

from ultralytics import YOLO

from picamera2 import Picamera2

from libcamera import Transform

import cv2

import time

import json

import math

import RPi.GPIO as GPIO

# ----------------------------

# MODEL / CAMERA SETTINGS

# ----------------------------

MODEL_PATH   = "/home/smartworkbench/best_n.pt"

CALIB_JSON   = "laser_grid_7x7_runtime_view.json"

FRAME_SIZE   = (480, 360)       # (W,H)

INFER_SIZE   = 320

# Global confidence threshold

CONF_THRESH_GLOBAL = 0.70

# Optional per-class threshold overrides (helps flickery classes like tape_measure)

# Leave empty {} if you don't want overrides.

CONF_THRESH_BY_NAME = {

   "tape_measure": 0.60,

}

# ----------------------------

# GPIO PINS

# ----------------------------

GPIO.setmode(GPIO.BCM)

PIN_TILT  = 23

PIN_PAN   = 24

PIN_LASER = 17

# Reed switches (1 drawer, 1 cabinet)

PIN_DRAWER  = 5

PIN_CABINET = 6

REED_DEBOUNCE_S = 0.05

REED_OPEN_IS_LOW = True  # typical wiring with PUD_UP: LOW means magnet present (closed) or open? choose below

# Define meaning:

# If you wire reed between GPIO and GND with pull-up:

# - magnet close -> switch closed -> GPIO reads LOW

# You likely want that to mean "CLOSED". If so, OPEN = HIGH.

# If you want the opposite, flip this logic below.

DRAWER_LOW_MEANS_CLOSED = True

CABINET_LOW_MEANS_CLOSED = True

# ----------------------------

# SERVO SETTINGS (SG90)

# ----------------------------

SERVO_HZ = 50

SERVO_UPDATE_PERIOD = 0.10   # seconds (10 Hz)

DEADBAND_DEG        = 1.0

PULSE_HOLD_TIME     = 0.08

PAN_MIN, PAN_MAX     = 0.0, 180.0

TILT_MIN, TILT_MAX   = 0.0, 180.0

# ----------------------------

# SMOOTHING / TIMEOUT

# ----------------------------

EMA_ALPHA         = 0.25

NO_DET_TIMEOUT    = 0.6

LOCK_LOST_TIMEOUT = 0.35

# UI presence persistence (prevents flicker in right panel)

PRESENCE_HOLD_S = 0.45

# Target hold (prevents bouncing in AUTO when multiple tools are present)

AUTO_SWITCH_HOLD_S   = 0.35   # must wait this long between target switches

AUTO_SWITCH_MARGIN   = 0.08   # new target must beat current by this margin

AUTO_TARGET_GRACE_S  = 0.25   # if target disappears briefly, keep it

# ----------------------------

# OPTIONAL GLOBAL TRIM

# ----------------------------

PAN_OFFSET_DEG  = 0.0

TILT_OFFSET_DEG = 0.0

# ----------------------------

# FIXED TOOL LIST (NO SHIFTING)

# ----------------------------

# Demo-friendly order (stable positions)

FIXED_TOOL_ORDER = [

   "wrench",

   "pliers",

   "screwdriver",

   "hammer",

   "tape_measure",

   "handsaw",

   "allen_keys",   # bonus class (often small)

]

# Search aliases (typing convenience)

SEARCH_ALIASES = {

   "allen": "allen_keys",

   "allen key": "allen_keys",

   "allen keys": "allen_keys",

   "hex": "allen_keys",

   "hex key": "allen_keys",

   "hex keys": "allen_keys",

   "tape": "tape_measure",

   "tape measure": "tape_measure",

   "measure": "tape_measure",

   "screw": "screwdriver",

   "driver": "screwdriver",

}

# ----------------------------

# UTILS

# ----------------------------

def clamp(v, lo, hi):

   return lo if v < lo else hi if v > hi else v

def angle_to_duty(angle):

   pulse_ms = 0.5 + (angle / 180.0) * 2.0

   return (pulse_ms / 20.0) * 100.0

def set_servo_angle(pwm, angle):

   duty = angle_to_duty(angle)

   pwm.ChangeDutyCycle(duty)

   time.sleep(PULSE_HOLD_TIME)

   pwm.ChangeDutyCycle(0)

def bilinear(u, v, tl, tr, bl, br):

   return (tl * (1-u) * (1-v) +

           tr * u     * (1-v) +

           bl * (1-u) * v     +

           br * u     * v)

def normalize_name(s):

   s = s.strip().lower()

   s = s.replace("-", " ").replace("_", " ")

   s = " ".join(s.split())

   return s

# ----------------------------

# GRID LOADING + PIECEWISE MAP

# ----------------------------

def load_grid(path):

   with open(path, "r") as f:

       d = json.load(f)

   u_list = [float(x) for x in d["u_list"]]

   v_list = [float(x) for x in d["v_list"]]

   pv = {}

   for p in d["points"]:

       key = (float(p["u"]), float(p["v"]))

       pv[key] = (float(p["pan"]), float(p["tilt"]))

   missing = []

   for v in v_list:

       for u in u_list:

           if (u, v) not in pv:

               missing.append((u, v))

   if missing:

       raise ValueError(f"Missing calibration points: {missing}")

   return u_list, v_list, pv

def find_cell(t, t_list):

   if t <= t_list[0]:

       return 0

   if t >= t_list[-1]:

       return len(t_list) - 2

   for i in range(len(t_list) - 1):

       if t_list[i] <= t <= t_list[i+1]:

           return i

   return len(t_list) - 2

def map_uv_to_pan_tilt(u, v, u_list, v_list, pv):

   u = clamp(u, 0.0, 1.0)

   v = clamp(v, 0.0, 1.0)

   i = find_cell(u, u_list)

   j = find_cell(v, v_list)

   u0, u1 = u_list[i], u_list[i+1]

   v0, v1 = v_list[j], v_list[j+1]

   uu = 0.0 if u1 == u0 else (u - u0) / (u1 - u0)

   vv = 0.0 if v1 == v0 else (v - v0) / (v1 - v0)

   uu = clamp(uu, 0.0, 1.0)

   vv = clamp(vv, 0.0, 1.0)

   pan_tl,  tilt_tl  = pv[(u0, v0)]

   pan_tr,  tilt_tr  = pv[(u1, v0)]

   pan_bl,  tilt_bl  = pv[(u0, v1)]

   pan_br,  tilt_br  = pv[(u1, v1)]

   pan  = bilinear(uu, vv, pan_tl,  pan_tr,  pan_bl,  pan_br)  + PAN_OFFSET_DEG

   tilt = bilinear(uu, vv, tilt_tl, tilt_tr, tilt_bl, tilt_br) + TILT_OFFSET_DEG

   pan  = clamp(pan,  PAN_MIN,  PAN_MAX)

   tilt = clamp(tilt, TILT_MIN, TILT_MAX)

   return pan, tilt

def map_xy_to_pan_tilt(x, y, frame_w, frame_h, u_list, v_list, pv):

   u = clamp(x / float(frame_w), 0.0, 1.0)

   v = clamp(y / float(frame_h), 0.0, 1.0)

   return map_uv_to_pan_tilt(u, v, u_list, v_list, pv)

# ----------------------------

# UI HELPERS

# ----------------------------

def blend_rect(img, x1, y1, x2, y2, color_bgr, alpha):

   x1 = max(0, min(img.shape[1], int(x1)))

   x2 = max(0, min(img.shape[1], int(x2)))

   y1 = max(0, min(img.shape[0], int(y1)))

   y2 = max(0, min(img.shape[0], int(y2)))

   if x2 <= x1 or y2 <= y1:

       return

   overlay = img.copy()

   cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)

   cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def conf_color(conf):

   conf = clamp(conf, 0.0, 1.0)

   if conf >= 0.85:

       return (60, 210, 90)     # green

   if conf >= 0.70:

       return (80, 190, 220)    # cyan

   if conf >= 0.55:

       return (80, 160, 240)    # blue-ish

   return (90, 90, 95)         # gray

def draw_crosshair(img, x, y, laser_on):

   h, w = img.shape[:2]

   x = int(clamp(x, 0, w - 1))

   y = int(clamp(y, 0, h - 1))

   size = 42

   thickness = 2

   color = (0, 235, 255)  # bright cyan

   cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)

   cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

   cv2.circle(img, (x, y), 10, color, 2, cv2.LINE_AA)

   cv2.circle(img, (x, y), 26, (255, 255, 255), 1, cv2.LINE_AA)

   if laser_on:

       t = time.time()

       pulse = (math.sin(t * 7.0) + 1.0) / 2.0

       rad = 6 + int(6 * pulse)

       cv2.circle(img, (x, y), rad, (60, 210, 90), -1, cv2.LINE_AA)

def draw_status_bar(img, mode, locked_name, laser_on, current_pan, current_tilt, fps,

                   drawer_open, cabinet_open):

   h, w = img.shape[:2]

   bar_h = 64

   blend_rect(img, 0, 0, w, bar_h, (18, 18, 20), 0.80)

   # MODE

   cv2.putText(img, mode, (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 235, 255), 2, cv2.LINE_AA)

   # LOCK

   if mode == "LOCK":

       cv2.putText(img, f"LOCK: {locked_name}", (150, 42),

                   cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)

   else:

       cv2.putText(img, "LOCK: none", (150, 42),

                   cv2.FONT_HERSHEY_SIMPLEX, 0.72, (150, 150, 150), 1, cv2.LINE_AA)

   # LASER indicator

   lx = 520

   cv2.circle(img, (lx, 32), 10, (60, 210, 90) if laser_on else (70, 70, 70),

              -1 if laser_on else 2, cv2.LINE_AA)

   cv2.putText(img, "LASER", (lx + 18, 40),

               cv2.FONT_HERSHEY_SIMPLEX, 0.65,

               (235, 235, 235) if laser_on else (150, 150, 150), 1, cv2.LINE_AA)

   # PAN/TILT

   cv2.putText(img, f"pan={current_pan:.0f}Â°  tilt={current_tilt:.0f}Â°", (lx + 120, 40),

               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

   # DRAWER / CABINET

   dx = lx + 360

   drawer_col = (60, 210, 90) if drawer_open else (110, 110, 110)

   cab_col    = (60, 210, 90) if cabinet_open else (110, 110, 110)

   cv2.putText(img, "DRAWER:", (dx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

   cv2.putText(img, "OPEN" if drawer_open else "CLOSED", (dx + 92, 40),

               cv2.FONT_HERSHEY_SIMPLEX, 0.62, drawer_col, 2 if drawer_open else 1, cv2.LINE_AA)

   cv2.putText(img, "CAB:", (dx + 190, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

   cv2.putText(img, "OPEN" if cabinet_open else "CLOSED", (dx + 240, 40),

               cv2.FONT_HERSHEY_SIMPLEX, 0.62, cab_col, 2 if cabinet_open else 1, cv2.LINE_AA)

   # FPS (right)

   fps_txt = f"{fps:.1f} FPS"

   tw, _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]

   cv2.putText(img, fps_txt, (w - tw - 22, 40),

               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

def draw_tool_panel(img, fixed_tools, selected_idx, locked_name, mode, tool_state):

   """

   fixed_tools: list of canonical tool names (fixed order)

   tool_state: dict name -> {present(bool), conf(float), last_seen(float)}

   """

   h, w = img.shape[:2]

   panel_w = 420

   x = w - panel_w - 22

   y = 86

   panel_h = min(h - y - 84, 520)

   blend_rect(img, x, y, x + panel_w, y + panel_h, (18, 18, 20), 0.74)

   cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (70, 70, 70), 1)

   cv2.putText(img, "Tools", (x + 16, y + 34),

               cv2.FONT_HERSHEY_SIMPLEX, 0.92, (235, 235, 235), 2, cv2.LINE_AA)

   # Controls hint

   cv2.putText(img, "TAB=select  ENTER=lock  A=auto  S=search", (x + 16, y + 58),

               cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 170, 170), 1, cv2.LINE_AA)

   row_h = 52

   base_y = y + 92

   for i, name in enumerate(fixed_tools):

       st = tool_state.get(name, {"present": False, "conf": 0.0, "last_seen": 0.0})

       present = bool(st["present"])

       conf = float(st["conf"])

       yy = base_y + i * row_h

       if yy + 26 > y + panel_h - 10:

           break

       is_sel = (i == selected_idx)

       is_locked = (mode == "LOCK" and locked_name == name)

       # Row background

       if is_locked:

           blend_rect(img, x + 10, yy - 30, x + panel_w - 10, yy + 18, (0, 200, 220), 0.25)

           cv2.rectangle(img, (x + 10, yy - 30), (x + panel_w - 10, yy + 18), (0, 235, 255), 2)

       elif is_sel:

           blend_rect(img, x + 10, yy - 30, x + panel_w - 10, yy + 18, (60, 60, 68), 0.70)

           cv2.rectangle(img, (x + 10, yy - 30), (x + panel_w - 10, yy + 18), (140, 140, 150), 1)

       else:

           blend_rect(img, x + 10, yy - 30, x + panel_w - 10, yy + 18, (35, 35, 38), 0.65)

       # Name style

       name_col = (245, 245, 245) if present else (130, 130, 130)

       cv2.putText(img, name, (x + 18, yy),

                   cv2.FONT_HERSHEY_SIMPLEX, 0.78, name_col, 2 if present else 1, cv2.LINE_AA)

       # Confidence bar (only if present)

       bar_x1 = x + 18

       bar_y1 = yy + 10

       bar_w = panel_w - 36

       bar_hh = 8

       cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_hh), (55, 55, 60), -1)

       if present:

           fill = int(bar_w * clamp(conf, 0.0, 1.0))

           cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + fill, bar_y1 + bar_hh), conf_color(conf), -1)

       cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_hh), (80, 80, 90), 1)

       # Conf text right

       conf_txt = f"{conf:.2f}" if present else "--"

       tw, _ = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]

       cv2.putText(img, conf_txt, (x + panel_w - tw - 16, yy),

                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 190, 190), 1, cv2.LINE_AA)

def draw_footer(img):

   h, w = img.shape[:2]

   blend_rect(img, 0, h - 44, w, h, (18, 18, 20), 0.74)

   txt = "ESC=quit   TAB=select   ENTER=lock   A=auto   S=search"

   cv2.putText(img, txt, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1, cv2.LINE_AA)

def draw_notification(img, message, level="info"):

   if not message:

       return

   h, w = img.shape[:2]

   y = h - 78

   if level == "warn":

       col = (0, 200, 220)

   elif level == "error":

       col = (60, 60, 220)

   else:

       col = (120, 120, 120)

   scale = 0.82

   th = 2

   (tw, thh), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, th)

   x1 = (w - tw) // 2 - 20

   x2 = (w + tw) // 2 + 20

   y1 = y - 34

   y2 = y + 12

   blend_rect(img, x1, y1, x2, y2, (18, 18, 20), 0.84)

   cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

   cv2.putText(img, message, (x1 + 18, y),

               cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), th, cv2.LINE_AA)

def draw_search_box(img, text):

   h, w = img.shape[:2]

   box_w = min(720, w - 60)

   x1 = (w - box_w) // 2

   y1 = 100

   x2 = x1 + box_w

   y2 = y1 + 70

   blend_rect(img, x1, y1, x2, y2, (18, 18, 20), 0.88)

   cv2.rectangle(img, (x1, y1), (x2, y2), (0, 235, 255), 2)

   cv2.putText(img, "Search tool:", (x1 + 16, y1 + 30),

               cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 1, cv2.LINE_AA)

   cv2.putText(img, text + ("_" if int(time.time() * 2) % 2 == 0 else ""), (x1 + 160, y1 + 30),

               cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)

   cv2.putText(img, "ENTER=lock   BACKSPACE=delete   ESC=cancel", (x1 + 16, y1 + 58),

               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

# ----------------------------

# FULLSCREEN FILL (scale + crop)

# ----------------------------

def resize_to_fill(img, out_w, out_h):

   """Scale image to cover (out_w,out_h) then center-crop."""

   h, w = img.shape[:2]

   if w <= 0 or h <= 0:

       return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

   scale = max(out_w / float(w), out_h / float(h))

   nw = int(w * scale)

   nh = int(h * scale)

   resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

   x0 = max(0, (nw - out_w) // 2)

   y0 = max(0, (nh - out_h) // 2)

   cropped = resized[y0:y0 + out_h, x0:x0 + out_w]

   if cropped.shape[1] != out_w or cropped.shape[0] != out_h:

       cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

   return cropped

# ----------------------------

# REED SWITCH DEBOUNCE

# ----------------------------

class DebouncedInput:

   def __init__(self, pin, pull=GPIO.PUD_UP, debounce_s=0.05):

       self.pin = pin

       GPIO.setup(pin, GPIO.IN, pull_up_down=pull)

       self.debounce_s = debounce_s

       self._last_raw = GPIO.input(pin)

       self._stable = self._last_raw

       self._last_change = time.time()

   def read(self):

       raw = GPIO.input(self.pin)

       now = time.time()

       if raw != self._last_raw:

           self._last_raw = raw

           self._last_change = now

       if (now - self._last_change) >= self.debounce_s and self._stable != raw:

           self._stable = raw

       return self._stable

def reed_state_to_open(stable_level, low_means_closed=True):

   # stable_level: GPIO.HIGH or GPIO.LOW

   if low_means_closed:

       return (stable_level == GPIO.HIGH)  # HIGH -> open

   else:

       return (stable_level == GPIO.LOW)   # LOW -> open

# ----------------------------

# SETUP GPIO OUTPUTS

# ----------------------------

GPIO.setup(PIN_PAN, GPIO.OUT)

GPIO.setup(PIN_TILT, GPIO.OUT)

GPIO.setup(PIN_LASER, GPIO.OUT)

GPIO.output(PIN_LASER, GPIO.LOW)

pwm_pan  = GPIO.PWM(PIN_PAN, SERVO_HZ)

pwm_tilt = GPIO.PWM(PIN_TILT, SERVO_HZ)

pwm_pan.start(0)

pwm_tilt.start(0)

drawer_in  = DebouncedInput(PIN_DRAWER,  pull=GPIO.PUD_UP, debounce_s=REED_DEBOUNCE_S)

cabinet_in = DebouncedInput(PIN_CABINET, pull=GPIO.PUD_UP, debounce_s=REED_DEBOUNCE_S)

# ----------------------------

# LOAD MODEL

# ----------------------------

model = YOLO(MODEL_PATH)

def class_name_from_id(cls_id: int) -> str:

   if isinstance(model.names, dict):

       return str(model.names.get(cls_id, cls_id))

   return str(model.names[cls_id])

# Build mapping canonical_name -> cls_id (by normalizing model names)

name_to_id = {}

for k, v in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names)):

   nm = normalize_name(str(v))

   nm = nm.replace(" ", "_")

   name_to_id[nm] = int(k)

# Ensure fixed tools exist in model

for nm in FIXED_TOOL_ORDER:

   if nm not in name_to_id:

       print(f"[WARN] Tool '{nm}' not found in model.names (normalized). Check your class naming.")

# ----------------------------

# LOAD GRID CALIBRATION

# ----------------------------

u_list, v_list, pv = load_grid(CALIB_JSON)

print(f"[OK] Loaded grid calibration {CALIB_JSON} ({len(u_list)}x{len(v_list)} = {len(pv)} points)")

# ----------------------------

# CAMERA SETUP

# ----------------------------

picam2 = Picamera2()

config = picam2.create_video_configuration(

   main={"size": FRAME_SIZE, "format": "RGB888"},

   transform=Transform(hflip=1, vflip=1)  # MUST match calibration orientation

)

picam2.configure(config)

picam2.start()

time.sleep(1.0)

W, H = FRAME_SIZE

# ----------------------------

# FULLSCREEN WINDOW SETUP

# ----------------------------

WINDOW_NAME = "F.O.R.G.E. Demo 2"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# We'll dynamically learn the actual fullscreen pixel size after first show

SCREEN_W, SCREEN_H = 1920, 1080

screen_size_locked = False

# Start at center

center_pan, center_tilt = map_xy_to_pan_tilt(W/2.0, H/2.0, W, H, u_list, v_list, pv)

current_pan  = clamp(center_pan,  PAN_MIN,  PAN_MAX)

current_tilt = clamp(center_tilt, TILT_MIN, TILT_MAX)

set_servo_angle(pwm_pan, current_pan)

set_servo_angle(pwm_tilt, current_tilt)

print("Started. AUTO by default. TAB select, ENTER lock, A auto, S search, ESC quit.")

# ----------------------------

# STATE

# ----------------------------

ema_x = None

ema_y = None

last_det_time = 0.0

last_servo_update = 0.0

mode = "AUTO"             # "AUTO" or "LOCK"

selected_idx = 0          # always valid in fixed list

locked_name = None        # canonical name (e.g., "wrench")

locked_cls_id = None      # int

last_locked_seen_time = 0.0

# AUTO target hold state

auto_target_cls_id = None

auto_target_last_seen = 0.0

auto_target_last_switch = 0.0

auto_target_conf = 0.0

# Toast notification

toast_msg = ""

toast_level = "info"

toast_until = 0.0

def toast(msg, level="info", seconds=1.2):

   global toast_msg, toast_level, toast_until

   toast_msg = msg

   toast_level = level

   toast_until = time.time() + seconds

# FPS smoothing

last_time = time.time()

fps = 0.0

# Tool presence state for fixed UI

tool_state = {nm: {"present": False, "conf": 0.0, "last_seen": 0.0} for nm in FIXED_TOOL_ORDER}

# Reed switch state tracking

prev_drawer_open = None

prev_cabinet_open = None

# Search mode

search_active = False

search_text = ""

# ----------------------------

# MAIN LOOP

# ----------------------------

try:

   while True:

       frame = picam2.capture_array()  # RGB

       now = time.time()

       # FPS

       dt = now - last_time

       if dt > 0:

           inst = 1.0 / dt

           fps = inst if fps <= 0 else (0.85 * fps + 0.15 * inst)

       last_time = now

       # Reed switches

       drawer_level = drawer_in.read()

       cabinet_level = cabinet_in.read()

       drawer_open = reed_state_to_open(drawer_level, low_means_closed=DRAWER_LOW_MEANS_CLOSED)

       cabinet_open = reed_state_to_open(cabinet_level, low_means_closed=CABINET_LOW_MEANS_CLOSED)

       if prev_drawer_open is None:

           prev_drawer_open = drawer_open

       if prev_cabinet_open is None:

           prev_cabinet_open = cabinet_open

       if drawer_open != prev_drawer_open:

           toast("Drawer opened" if drawer_open else "Drawer closed", "info", 1.1)

           prev_drawer_open = drawer_open

       if cabinet_open != prev_cabinet_open:

           toast("Cabinet opened" if cabinet_open else "Cabinet closed", "info", 1.1)

           prev_cabinet_open = cabinet_open

       # Inference (use global conf; we'll filter per-class later)

       results = model(frame, imgsz=INFER_SIZE, conf=0.25, verbose=False)

       r = results[0]

       # Collect best detection per class (for UI + targeting)

       best_by_cls = {}  # cls_id -> (conf, cx, cy, name)

       best_overall = None  # (conf, cx, cy, name, cls_id)

       if r.boxes is not None and len(r.boxes):

           for box in r.boxes:

               conf = float(box.conf[0])

               cls_id = int(box.cls[0])

               name_raw = class_name_from_id(cls_id)

               name_norm = normalize_name(str(name_raw)).replace(" ", "_")

               # Per-class thresholding (demo polish)

               th = CONF_THRESH_BY_NAME.get(name_norm, CONF_THRESH_GLOBAL)

               if conf < th:

                   continue

               x1, y1, x2, y2 = box.xyxy[0].tolist()

               cx = (x1 + x2) / 2.0

               cy = (y1 + y2) / 2.0

               # Best overall

               if (best_overall is None) or (conf > best_overall[0]):

                   best_overall = (conf, cx, cy, name_norm, cls_id)

               # Best per class

               prev = best_by_cls.get(cls_id)

               if (prev is None) or (conf > prev[0]):

                   best_by_cls[cls_id] = (conf, cx, cy, name_norm)

       # Update UI presence with persistence

       for nm in FIXED_TOOL_ORDER:

           tool_state[nm]["present"] = False  # will be re-enabled if seen/persisted

       # Mark seen now

       for cls_id, (conf, cx, cy, nm) in best_by_cls.items():

           if nm in tool_state:

               tool_state[nm]["conf"] = conf

               tool_state[nm]["last_seen"] = now

               tool_state[nm]["present"] = True

       # Apply persistence

       for nm in FIXED_TOOL_ORDER:

           last_seen = tool_state[nm]["last_seen"]

           if (now - last_seen) <= PRESENCE_HOLD_S:

               # treat as present (even if not detected this exact frame)

               tool_state[nm]["present"] = True if last_seen > 0 else False

           else:

               tool_state[nm]["present"] = False

               tool_state[nm]["conf"] = 0.0

       # ----------------------------

       # TARGET SELECTION

       # ----------------------------

       best = None  # (conf, cx, cy, name_norm, cls_id)

       if mode == "LOCK" and locked_cls_id is not None:

           # LOCK: best among locked class only

           if locked_cls_id in best_by_cls:

               conf, cx, cy, nm = best_by_cls[locked_cls_id]

               best = (conf, cx, cy, nm, locked_cls_id)

       else:

           # AUTO with hysteresis

           if best_overall is not None:

               new_conf, new_cx, new_cy, new_nm, new_id = best_overall

               # If we have a current target, check if it's still around

               current_visible = False

               current_conf = 0.0

               current_cx = None

               current_cy = None

               current_nm = None

               if auto_target_cls_id is not None and auto_target_cls_id in best_by_cls:

                   current_visible = True

                   current_conf, current_cx, current_cy, current_nm = best_by_cls[auto_target_cls_id]

                   auto_target_last_seen = now

               else:

                   # allow grace if it disappeared briefly

                   if auto_target_cls_id is not None and (now - auto_target_last_seen) <= AUTO_TARGET_GRACE_S:

                       current_visible = False  # not in boxes, but still "held"

                   else:

                       auto_target_cls_id = None

               # Decide switch

               if auto_target_cls_id is None:

                   auto_target_cls_id = new_id

                   auto_target_conf = new_conf

                   auto_target_last_seen = now

                   auto_target_last_switch = now

               else:

                   # If current target visible, only switch if new beats by margin and hold time elapsed

                   can_switch_time = (now - auto_target_last_switch) >= AUTO_SWITCH_HOLD_S

                   if can_switch_time:

                       if (new_id != auto_target_cls_id) and (new_conf >= (auto_target_conf + AUTO_SWITCH_MARGIN)):

                           auto_target_cls_id = new_id

                           auto_target_conf = new_conf

                           auto_target_last_seen = now

                           auto_target_last_switch = now

                   # Update stored conf when visible

                   if current_visible:

                       auto_target_conf = current_conf

               # Output best target based on chosen id

               if auto_target_cls_id is not None and auto_target_cls_id in best_by_cls:

                   conf, cx, cy, nm = best_by_cls[auto_target_cls_id]

                   best = (conf, cx, cy, nm, auto_target_cls_id)

       # ----------------------------

       # LASER + SERVO CONTROL

       # ----------------------------

       laser_state = False

       if mode == "LOCK" and locked_cls_id is not None:

           if best is not None:

               last_locked_seen_time = now

               last_det_time = now

               GPIO.output(PIN_LASER, GPIO.HIGH)

               laser_state = True

           else:

               lost_for = now - last_locked_seen_time

               if lost_for > LOCK_LOST_TIMEOUT:

                   GPIO.output(PIN_LASER, GPIO.LOW)

                   laser_state = False

                   ema_x = None

                   ema_y = None

       else:

           # AUTO timeout

           if best is not None:

               last_det_time = now

               GPIO.output(PIN_LASER, GPIO.HIGH)

               laser_state = True

           else:

               if (now - last_det_time) > NO_DET_TIMEOUT:

                   GPIO.output(PIN_LASER, GPIO.LOW)

                   laser_state = False

                   ema_x = None

                   ema_y = None

       # If we have a target, update EMA + move servos

       if best is not None:

           conf, cx, cy, nm, cls_id = best

           if ema_x is None:

               ema_x, ema_y = cx, cy

           else:

               ema_x = EMA_ALPHA * cx + (1.0 - EMA_ALPHA) * ema_x

               ema_y = EMA_ALPHA * cy + (1.0 - EMA_ALPHA) * ema_y

           target_pan, target_tilt = map_xy_to_pan_tilt(ema_x, ema_y, W, H, u_list, v_list, pv)

           if (now - last_servo_update) >= SERVO_UPDATE_PERIOD:

               moved = False

               if abs(target_pan - current_pan) >= DEADBAND_DEG:

                   current_pan = target_pan

                   set_servo_angle(pwm_pan, current_pan)

                   moved = True

               if abs(target_tilt - current_tilt) >= DEADBAND_DEG:

                   current_tilt = target_tilt

                   set_servo_angle(pwm_tilt, current_tilt)

                   moved = True

               if moved:

                   last_servo_update = now

       # ----------------------------

       # RENDER

       # ----------------------------

       annotated = r.plot()  # Ultralytics returns BGR image with boxes

       # Determine fullscreen size once (after first render)

       if not screen_size_locked:

           # Show a temporary frame to force window geometry

           temp = cv2.resize(annotated, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

           cv2.imshow(WINDOW_NAME, temp)

           cv2.waitKey(1)

           try:

               rx, ry, rw, rh = cv2.getWindowImageRect(WINDOW_NAME)

               if rw > 200 and rh > 200:

                   SCREEN_W, SCREEN_H = rw, rh

                   screen_size_locked = True

           except Exception:

               pass

       display = resize_to_fill(annotated, SCREEN_W, SCREEN_H)

       # Crosshair on target (scaled by fill method)

       # Because we used resize_to_fill (scale+crop), we must map coordinates accordingly.

       if best is not None and ema_x is not None and ema_y is not None:

           # compute mapping from original (W,H) to display (SCREEN_W,SCREEN_H) with scale+crop

           scale = max(SCREEN_W / float(W), SCREEN_H / float(H))

           nw = W * scale

           nh = H * scale

           x0 = (nw - SCREEN_W) / 2.0

           y0 = (nh - SCREEN_H) / 2.0

           tx = ema_x * scale - x0

           ty = ema_y * scale - y0

           draw_crosshair(display, tx, ty, laser_state)

           label = f"{best[3]}  {best[0]:.2f}"

           cv2.putText(display, label, (int(tx + 18), int(ty - 18)),

                       cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)

       # UI overlay

       draw_status_bar(display, mode, locked_name if locked_name else "none",

                       laser_state, current_pan, current_tilt, fps,

                       drawer_open, cabinet_open)

       draw_tool_panel(display, FIXED_TOOL_ORDER, selected_idx, locked_name, mode, tool_state)

       draw_footer(display)

       # Toast

       if toast_msg and time.time() < toast_until:

           draw_notification(display, toast_msg, toast_level)

       # Search overlay

       if search_active:

           draw_search_box(display, search_text)

       cv2.imshow(WINDOW_NAME, display)

       key = cv2.waitKey(1) & 0xFF

       # ----------------------------

       # INPUT HANDLING

       # ----------------------------

       # ESC always exits search or quits

       if key == 27:

           if search_active:

               search_active = False

               search_text = ""

               toast("Search cancelled", "warn", 0.9)

           else:

               break

       if search_active:

           # ENTER confirms search

           if key in (10, 13):  # Enter

               q = normalize_name(search_text)

               canonical = None

               if q in SEARCH_ALIASES:

                   canonical = SEARCH_ALIASES[q]

               else:

                   # direct match to fixed names

                   q2 = q.replace(" ", "_")

                   if q2 in FIXED_TOOL_ORDER:

                       canonical = q2

               if canonical is None:

                   toast(f"Unknown tool: '{search_text}'", "warn", 1.3)

               else:

                   locked_name = canonical

                   locked_cls_id = name_to_id.get(canonical, None)

                   if locked_cls_id is None:

                       toast(f"Tool '{canonical}' not in model", "error", 1.4)

                       mode = "AUTO"

                       locked_name = None

                       locked_cls_id = None

                   else:

                       mode = "LOCK"

                       last_locked_seen_time = now

                       toast(f"LOCK: {locked_name}", "info", 1.0)

               search_active = False

               search_text = ""

               continue

           # BACKSPACE

           if key in (8, 127):

               search_text = search_text[:-1]

               continue

           # Accept normal characters (letters, numbers, space, underscore)

           if 32 <= key <= 126:

               ch = chr(key)

               if ch.isalnum() or ch in (" ", "_", "-"):

                   # keep it reasonable length

                   if len(search_text) < 22:

                       search_text += ch

               continue

           continue  # ignore other keys while searching

       # Not in search mode: normal controls

       if key == 9:  # TAB

           selected_idx = (selected_idx + 1) % len(FIXED_TOOL_ORDER)

       elif key in (10, 13):  # ENTER

           # lock selected tool

           nm = FIXED_TOOL_ORDER[selected_idx]

           cls_id = name_to_id.get(nm, None)

           if cls_id is None:

               toast(f"Tool '{nm}' not in model", "error", 1.4)

           else:

               mode = "LOCK"

               locked_name = nm

               locked_cls_id = cls_id

               last_locked_seen_time = now

               toast(f"LOCK: {locked_name}", "info", 1.0)

       elif key in (ord('a'), ord('A')):

           mode = "AUTO"

           locked_name = None

           locked_cls_id = None

           toast("MODE: AUTO", "info", 1.0)

       elif key in (ord('s'), ord('S')):

           search_active = True

           search_text = ""

           toast("Search: type tool name", "info", 1.0)

       # Optional: quick lock with L (lock/unlock toggle)

       elif key in (ord('l'), ord('L')):

           if mode == "LOCK":

               mode = "AUTO"

               locked_name = None

               locked_cls_id = None

               toast("MODE: AUTO", "info", 1.0)

           else:

               nm = FIXED_TOOL_ORDER[selected_idx]

               cls_id = name_to_id.get(nm, None)

               if cls_id is not None:

                   mode = "LOCK"

                   locked_name = nm

                   locked_cls_id = cls_id

                   last_locked_seen_time = now

                   toast(f"LOCK: {locked_name}", "info", 1.0)

finally:

   GPIO.output(PIN_LASER, GPIO.LOW)

   try:

       pwm_pan.ChangeDutyCycle(0)

       pwm_tilt.ChangeDutyCycle(0)

       pwm_pan.stop()

       pwm_tilt.stop()

   except Exception:

       pass

   GPIO.cleanup()

   cv2.destroyAllWindows()

   try:

       picam2.stop()

   except Exception:

       pass
 
