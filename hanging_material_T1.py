import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
video_path = "/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/1 hanging material.mp4"   # Update path if needed

# ROI (top-right portion of frame where hanging object is expected)
ROI_W_RATIO = 0.35   # rightmost 35% of width
ROI_H_RATIO = 0.50   # top 50% of height

# Contour filtering
MIN_CONTOUR_AREA_RATIO = 0.00005  # fraction of total frame area
ASPECT_RATIO_MIN = 1.2            # prefer tall shapes

# Oscillation detection thresholds
AMP_MIN_PIXELS = 8.0              # min horizontal amplitude
AMP_MIN_RATIO = 0.01              # or 1% of width
SIGN_CHANGE_MIN = 6               # minimum centroid direction changes
FLOW_SIGN_CHANGE_MIN = 8          # minimum flow direction changes

# Performance tuning
PROCESS_EVERY_NTH_FRAME = 2       # skip frames for speed


def main():
    assert os.path.exists(video_path), f"Video not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_x0 = int(width * (1 - ROI_W_RATIO))
    roi_y0 = 0
    roi_w  = int(width * ROI_W_RATIO)
    roi_h  = int(height * ROI_H_RATIO)

    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = "hanging_annotated.mp4"
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    prev_gray_roi = None
    centroid_x, centroid_y, frame_idx, mean_u, boxes = [], [], [], [], []

    min_contour_area = max(150, int((width * height) * MIN_CONTOUR_AREA_RATIO))
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % PROCESS_EVERY_NTH_FRAME != 0:
            vw.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y0:roi_y0 + roi_h, roi_x0:roi_x0 + roi_w]

        # Background subtraction
        fg = bg.apply(roi)
        fg = cv2.medianBlur(fg, 5)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chosen_box, chosen_cx, chosen_cy, best_score = None, None, None, -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = h / (w + 1e-6)
            top_nearness = 1.0 - (y / (roi_h + 1e-6))
            score = (0.5 * top_nearness) + (0.3 * min(ar / 2.0, 1.0)) + 0.2
            if ar >= ASPECT_RATIO_MIN and score > best_score:
                best_score = score
                chosen_box = (x, y, w, h)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    chosen_cx = int(M["m10"] / M["m00"])
                    chosen_cy = int(M["m01"] / M["m00"])
                else:
                    chosen_cx, chosen_cy = x + w // 2, y + h // 2

        # Dense optical flow
        u_mean = 0.0
        if prev_gray_roi is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_roi, roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            u = flow[..., 0]
            if chosen_box is not None:
                x, y, w, h = chosen_box
                u_crop = u[y:y + h, x:x + w]
                u_mean = float(np.nanmean(u_crop)) if u_crop.size > 0 else float(np.nanmean(u))
            else:
                u_mean = float(np.nanmean(u))
        prev_gray_roi = roi

        # Save tracking data
        if chosen_box is not None:
            centroid_x.append(roi_x0 + chosen_cx)
            centroid_y.append(roi_y0 + chosen_cy)
            boxes.append((roi_x0 + chosen_box[0], roi_y0 + chosen_box[1], chosen_box[2], chosen_box[3]))
        else:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)
            boxes.append(None)

        frame_idx.append(frame_counter)
        mean_u.append(u_mean)

        # Visualization
        fvis = frame.copy()
        cv2.rectangle(fvis, (roi_x0, roi_y0), (roi_x0 + roi_w, roi_y0 + roi_h), (255, 255, 255), 2)
        if chosen_box is not None:
            x, y, w, h = boxes[-1]
            cv2.rectangle(fvis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(fvis, (centroid_x[-1], centroid_y[-1]), 3, (0, 0, 255), -1)
            cv2.arrowedLine(
                fvis,
                (centroid_x[-1], centroid_y[-1]),
                (int(centroid_x[-1] + 15 * np.sign(u_mean)), centroid_y[-1]),
                (0, 255, 255), 2, tipLength=0.4
            )
        vw.write(fvis)

    cap.release()
    vw.release()

    # Analysis
    cx = np.array(centroid_x, dtype=float)
    valid = ~np.isnan(cx)
    disp = cx - cx[valid][0] if np.any(valid) else np.zeros_like(cx)

    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO * width)
    amp = np.nanmax(disp) - np.nanmin(disp) if len(disp) else 0.0

    dx = np.diff(disp)
    dx_mag = np.abs(dx)
    dx_sign = np.sign(dx)
    sig_mask = dx_mag > max(2.0, 0.003 * width)
    sign_changes = np.sum((dx_sign[1:] * dx_sign[:-1] < 0) & sig_mask[1:])

    mu = np.array(mean_u, dtype=float)
    mu_sign = np.sign(np.nan_to_num(mu))
    mu_changes = np.sum(mu_sign[1:] * mu_sign[:-1] < 0)

    oscillatory = (amp >= amp_thresh) and (sign_changes >= SIGN_CHANGE_MIN or mu_changes >= FLOW_SIGN_CHANGE_MIN)

    # Plot displacement signal
    # plt.figure(figsize=(10, 4))
    # plt.title("Horizontal centroid displacement (pixels)")
    # plt.xlabel("Processed frame index")
    # plt.ylabel("Δx from first detection")
    # plt.plot(frame_idx, disp)
    # plt.tight_layout()
    # plot_path = "hanging_signal.png"
    # plt.savefig(plot_path)
    # plt.close()

    # Save summary
#     summary = f"""Video: {os.path.basename(video_path)}
# Frame size: {width}x{height}  FPS: {fps:.2f}  Frames processed: ~{len(frame_idx)} / {total}
# ROI (top-right): x={roi_x0}, y={roi_y0}, w={roi_w}, h={roi_h}
# Detected amplitude (Δx): {amp:.2f} px  | Threshold: {amp_thresh:.2f} px
# Direction changes (centroid Δx): {int(sign_changes)}
# Flow sign changes (mean u in ROI): {int(mu_changes)}
# Oscillatory verdict: {'YES' if oscillatory else 'NO'}
# """
#     with open("hanging_analysis_summary.txt", "w") as f:
#         f.write(summary)

    # print(summary)
    print("\n✅ Outputs generated:")
    print(f" - Annotated video: {out_video_path}")
    # print(f" - Displacement plot: {plot_path}")
    print(f" - Summary text: hanging_analysis_summary.txt")


if __name__ == "__main__":
    main()
