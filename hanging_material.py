import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# -------------------------- CONFIG --------------------------
VIDEO_PATH = "/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/1 hanging material.mp4"  # change or pass via CLI if you like
OUTPUT_DIR = "./hanging_outputs"

UPPER_FRACTION = 0.90     # analyze top 90% of frame
PROCESS_EVERY = 2         # process every Nth frame for speed

# Candidate filtering (tune for your footage)
MIN_AREA_RATIO = 0.00005  # relative to full frame area
MIN_ASPECT = 1.2          # tall-ish blobs (h/w)
MAX_ASPECT = 10.0         # avoid absurdly tall artifacts
MIN_HEIGHT_RATIO = 0.05   # vertical extent >= 5% of frame height
MAX_WIDTH_RATIO = 0.25    # cap width at <= 25% of frame width (thin-ish)

# Association
MAX_CENTROID_DIST = 60    # px, link detections to existing tracks

# Oscillation decision
AMP_MIN_PIXELS = 8.0
AMP_MIN_RATIO  = 0.01     # 1% of width (use max of pixels/ratio)
MIN_DIR_CHANGES = 6       # centroid Δx sign flips (with magnitude)
MIN_FLOW_FLIPS  = 8       # mean u sign flips

# Visualization
DRAW_FLOW_ARROW = True

# ------------------------------------------------------------

class Track:
    _next_id = 1
    def __init__(self, cx, cy, box, frame_idx):
        self.id = Track._next_id; Track._next_id += 1
        self.cx_hist = deque([cx], maxlen=5000)
        self.cy_hist = deque([cy], maxlen=5000)
        self.t_hist  = deque([frame_idx], maxlen=5000)
        self.box_hist = deque([box], maxlen=5000)
        self.flow_u = deque([], maxlen=5000)
        self.missed = 0
        self.alive  = True

    def update(self, cx, cy, box, frame_idx, u_mean=None):
        self.cx_hist.append(cx); self.cy_hist.append(cy)
        self.t_hist.append(frame_idx); self.box_hist.append(box)
        if u_mean is not None: self.flow_u.append(u_mean)
        self.missed = 0

    def age(self): return len(self.t_hist)

def iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa, ya = max(x1,x2), max(y1,y2)
    xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xb-xa), max(0, yb-ya)
    inter = iw*ih
    if inter == 0: return 0.0
    a1 = w1*h1; a2 = w2*h2
    return inter / float(a1 + a2 - inter)

def decide_oscillation(track, frame_w):
    cx = np.array(track.cx_hist, dtype=float)
    if len(cx) < 5: return False, {"amp":0,"dir_changes":0,"flow_changes":0}
    disp = cx - cx[0]
    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO*frame_w)
    amp = np.nanmax(disp) - np.nanmin(disp) if disp.size else 0.0

    dx = np.diff(disp)
    dx_mag = np.abs(dx)
    dx_sign = np.sign(dx)
    sig_mask = dx_mag > max(2.0, 0.003*frame_w)
    dir_changes = int(np.sum((dx_sign[1:]*dx_sign[:-1] < 0) & sig_mask[1:]))

    flow_changes = 0
    if len(track.flow_u) >= 3:
        mu = np.array(track.flow_u, dtype=float)
        mu = np.nan_to_num(mu, nan=0.0)
        mu_s = np.sign(mu)
        flow_changes = int(np.sum(mu_s[1:]*mu_s[:-1] < 0))

    verdict = (amp >= amp_thresh) and (dir_changes >= MIN_DIR_CHANGES or flow_changes >= MIN_FLOW_FLIPS)
    return verdict, {"amp":amp, "dir_changes":dir_changes, "flow_changes":flow_changes}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_track(track, out_png, title="Δx over frames"):
    cx = np.array(track.cx_hist, dtype=float)
    t  = np.array(track.t_hist, dtype=int)
    if len(cx) < 2: return
    disp = cx - cx[0]
    plt.figure(figsize=(10,4))
    plt.title(title + f" (Track #{track.id})")
    plt.xlabel("Frame idx (processed)"); plt.ylabel("Δx (px)")
    plt.plot(t, disp)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Analyze upper 90% of the frame
    roi_y0, roi_h = 0, int(H * UPPER_FRACTION)

    # Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = os.path.join(OUTPUT_DIR, "hanging_annotated.mp4")
    vw = cv2.VideoWriter(out_vid, fourcc, FPS, (W, H))

    bg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    tracks = []
    frame_idx = 0
    prev_roi_gray = None

    min_area = max(150, int(MIN_AREA_RATIO * W * H))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Write-through for skipped frames to keep durations aligned
        if frame_idx % PROCESS_EVERY != 0:
            vw.write(frame); continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = gray[roi_y0:roi_y0+roi_h, 0:W]

        # Foreground mask (upper band)
        fg = bg.apply(roi)
        fg = cv2.medianBlur(fg, 5)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute dense flow for mean horizontal motion (optional but robust)
        u = None
        if prev_roi_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_roi_gray, roi, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            u = flow[...,0]
        prev_roi_gray = roi

        # Build detection list
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            # restore full-frame coords
            X,Y = x, y + roi_y0

            # shape filters
            if h < H * MIN_HEIGHT_RATIO: continue
            if w > W * MAX_WIDTH_RATIO: continue
            aspect = h / (w + 1e-6)
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue

            # centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_rel = int(M["m10"]/M["m00"]); cy_rel = int(M["m01"]/M["m00"])
            else:
                cx_rel, cy_rel = x + w//2, y + h//2
            cx, cy = cx_rel, cy_rel + roi_y0

            # mean horizontal flow inside box (if available)
            u_mean = 0.0
            if u is not None:
                ux = u[y:y+h, x:x+w]
                if ux.size > 0: u_mean = float(np.nanmean(ux))
            dets.append(((X,Y,w,h), (cx,cy), u_mean))

        # Associate detections to tracks (greedy by centroid distance)
        assigned = set()
        for tr in tracks:
            if not tr.alive: continue
            tr.missed += 1

        for i, (box, (cx,cy), u_mean) in enumerate(dets):
            best, best_d, best_idx = None, 1e9, -1
            for ti, tr in enumerate(tracks):
                if not tr.alive or ti in assigned: continue
                px, py = tr.cx_hist[-1], tr.cy_hist[-1]
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d, best, best_idx = d, tr, ti
            if best is not None and best_d <= MAX_CENTROID_DIST:
                best.update(cx, cy, box, frame_idx, u_mean)
                best.missed = 0
                assigned.add(best_idx)
            else:
                tr = Track(cx, cy, box, frame_idx)
                tr.update(cx, cy, box, frame_idx, u_mean)
                tracks.append(tr)
                assigned.add(len(tracks)-1)

        # Kill stale tracks
        for tr in tracks:
            if tr.missed > 10: tr.alive = False

        # Visualization
        vis = frame.copy()
        # draw upper band
        cv2.rectangle(vis, (0, roi_y0), (W, roi_y0+roi_h), (255,255,255), 2)

        for tr in tracks:
            if not tr.alive or len(tr.box_hist)==0: continue
            x,y,w,h = tr.box_hist[-1]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cx,cy = int(tr.cx_hist[-1]), int(tr.cy_hist[-1])
            cv2.circle(vis, (cx,cy), 3, (0,0,255), -1)
            if DRAW_FLOW_ARROW and len(tr.flow_u)>0:
                u_mean = tr.flow_u[-1]
                cv2.arrowedLine(vis, (cx,cy),
                                (int(cx + 15*np.sign(u_mean)), cy),
                                (0,255,255), 2, tipLength=0.4)
            cv2.putText(vis, f"ID{tr.id}", (x, max(0,y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv2.LINE_AA)

        vw.write(vis)

    cap.release(); vw.release()

    # Decide best oscillating track
    ensure_dir(OUTPUT_DIR)
    summary_lines = []
    best_track, best_score = None, -1

    # for tr in tracks:
    #     if tr.age() < 8: continue
    #     verdict, stats = decide_oscillation(tr, W)
    #     score = stats["amp"] + 5*stats["dir_changes"] + 3*stats["flow_changes"]
    #     tag = "YES" if verdict else "NO"
    #     summary_lines.append(
    #         f"Track #{tr.id}: age={tr.age():3d}  amp={stats['amp']:.1f}  "
    #         f"dir_changes={stats['dir_changes']:2d}  flow_changes={stats['flow_changes']:2d}  OSC={tag}"
    #     )
    #     # save individual plot
    #     plot_path = os.path.join(OUTPUT_DIR, f"track_{tr.id}_dx.png")
    #     plot_track(tr, plot_path, "Horizontal centroid displacement")
    #     if verdict and score > best_score:
    #         best_score, best_track = score, tr

    # Combined verdict
    # overall = "YES" if best_track is not None else "NO"
    # head = [
    #     f"Video: {os.path.basename(VIDEO_PATH)}",
    #     f"Frame: {W}x{H}  FPS:{FPS:.2f}  Total:{N}",
    #     f"Analyzed band: top {int(UPPER_FRACTION*100)}% (y=[0, {int(UPPER_FRACTION*H)}])",
    #     f"Overall oscillation: {overall}",
    #     ""
    # ]
    # report = "\n".join(head + summary_lines)
    # with open(os.path.join(OUTPUT_DIR, "hanging_summary.txt"), "w") as f:
    #     f.write(report)

    # print(report)
    # print("\nOutputs:")
    # print(f" - Annotated video: {out_vid}")
    # print(f" - Summary: {os.path.join(OUTPUT_DIR,'hanging_summary.txt')}")
    # print(f" - Per-track plots: {os.path.join(OUTPUT_DIR,'track_*_dx.png')}")

if __name__ == "__main__":
    main()
