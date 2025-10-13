import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
VIDEO_PATH = "/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/5 hanging material.mp4"     # update if needed
OUTPUT_DIR = "./hanging_outputs"

UPPER_FRACTION = 0.90    # analyze top 90% of frame
PROCESS_EVERY = 2        # skip frames for speed

# Detection filters
MIN_ASPECT_RATIO = 1.2        # tall shape filter
MIN_HEIGHT_RATIO = 0.50       # object height ≥ 50% of frame
MIN_AREA_RATIO = 0.0005       # relative area threshold
MAX_WIDTH_RATIO = 0.4         # reject extremely wide objects

# Oscillation logic
AMP_MIN_PIXELS = 8.0
AMP_MIN_RATIO = 0.01
MIN_DIR_CHANGES = 6
MIN_FLOW_FLIPS = 8
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_y0, roi_h = 0, int(H * UPPER_FRACTION)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = os.path.join(OUTPUT_DIR, "hanging_annotated.mp4")
    vw = cv2.VideoWriter(out_video, fourcc, FPS, (W, H))

    bg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    prev_roi_gray = None
    cx_hist, frame_idx, mean_u_hist = [], [], []
    box_hist = []

    min_area = max(200, int(MIN_AREA_RATIO * W * H))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % PROCESS_EVERY != 0:
            vw.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y0:roi_y0 + roi_h, 0:W]

        # Foreground mask
        fg = bg.apply(roi)
        fg = cv2.medianBlur(fg, 5)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dense flow for horizontal motion
        u = None
        if prev_roi_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_roi_gray, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            u = flow[..., 0]
        prev_roi_gray = roi

        # Choose main tall object
        chosen_box = None
        chosen_cx, chosen_cy = None, None
        chosen_u_mean = 0.0
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            X, Y = x, y + roi_y0

            aspect = h / (w + 1e-6)
            if aspect < MIN_ASPECT_RATIO or h < H * MIN_HEIGHT_RATIO or w > W * MAX_WIDTH_RATIO:
                continue

            score = area * aspect  # area * tallness

            if score > best_score:
                best_score = score
                chosen_box = (X, Y, w, h)

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx_rel = int(M["m10"] / M["m00"])
                    cy_rel = int(M["m01"] / M["m00"])
                else:
                    cx_rel, cy_rel = x + w // 2, y + h // 2

                chosen_cx, chosen_cy = cx_rel, cy_rel + roi_y0

                if u is not None:
                    ux = u[y:y + h, x:x + w]
                    if ux.size > 0:
                        chosen_u_mean = float(np.nanmean(ux))

        # Record if found
        if chosen_box is not None:
            cx_hist.append(chosen_cx)
            frame_idx.append(frame_count)
            mean_u_hist.append(chosen_u_mean)
            box_hist.append(chosen_box)

        # Draw annotation
        vis = frame.copy()
        cv2.rectangle(vis, (0, roi_y0), (W, roi_y0 + roi_h), (255, 255, 255), 2)
        if chosen_box is not None:
            x, y, w, h = chosen_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (chosen_cx, chosen_cy), 4, (0, 0, 255), -1)
            cv2.arrowedLine(vis, (chosen_cx, chosen_cy),
                            (int(chosen_cx + 15 * np.sign(chosen_u_mean)), chosen_cy),
                            (0, 255, 255), 2, tipLength=0.4)

        vw.write(vis)

    cap.release()
    vw.release()

    # ======= Oscillation Decision =======
    cx = np.array(cx_hist, dtype=float)
    if len(cx) < 5:
        print("Not enough frames detected.")
        return

    disp = cx - cx[0]
    amp = np.nanmax(disp) - np.nanmin(disp)
    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO * W)

    dx = np.diff(disp)
    dx_mag = np.abs(dx)
    dx_sign = np.sign(dx)
    sig_mask = dx_mag > max(2.0, 0.003 * W)
    dir_changes = np.sum((dx_sign[1:] * dx_sign[:-1] < 0) & sig_mask[1:])

    mu = np.array(mean_u_hist, dtype=float)
    mu = np.nan_to_num(mu, nan=0.0)
    mu_sign = np.sign(mu)
    flow_changes = np.sum(mu_sign[1:] * mu_sign[:-1] < 0)

    oscillatory = (amp >= amp_thresh) and (dir_changes >= MIN_DIR_CHANGES or flow_changes >= MIN_FLOW_FLIPS)

    # # ======= Save outputs =======
    # disp_plot = os.path.join(OUTPUT_DIR, "hanging_displacement.png")
    # plt.figure(figsize=(10, 4))
    # plt.title("Horizontal centroid displacement (Δx)")
    # plt.xlabel("Frame")
    # plt.ylabel("Δx (pixels)")
    # plt.plot(frame_idx, disp)
    # plt.tight_layout()
    # plt.savefig(disp_plot)
    # plt.close()

    # summary_txt = os.path.join(OUTPUT_DIR, "hanging_summary.txt")
    # with open(summary_txt, "w") as f:
    #     f.write(f"Video: {os.path.basename(VIDEO_PATH)}\n")
    #     f.write(f"Resolution: {W}x{H} | FPS: {FPS:.2f} | Frames: {N}\n")
    #     f.write(f"Analyzed Region: Top {int(UPPER_FRACTION * 100)}%\n")
    #     f.write(f"Detected amplitude Δx: {amp:.2f} px | Threshold: {amp_thresh:.2f} px\n")
    #     f.write(f"Direction changes: {dir_changes}\n")
    #     f.write(f"Flow sign changes: {flow_changes}\n")
    #     f.write(f"Oscillation verdict: {'YES' if oscillatory else 'NO'}\n")

    # print("\n✅ Analysis complete:")
    # print(f" - Annotated video: {out_video}")
    # print(f" - Displacement plot: {disp_plot}")
    # print(f" - Summary: {summary_txt}")
    # print(f" - Oscillatory: {'YES' if oscillatory else 'NO'}")

if __name__ == "__main__":
    main()
