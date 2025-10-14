import cv2
import os
import numpy as np

OUTPUT_DIR = "./fast_hanging_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPPER_FRACTION = 0.90
PROCESS_EVERY = 2
AMP_MIN_PIXELS = 8.0
AMP_MIN_RATIO = 0.01
MIN_DIR_CHANGES = 6

def main(video_number):
    # video_path = f"/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/{video_number} hanging material.mp4"
    video_path = f"/data1/yasir/Data/Abnormal Behavior2/Abnormal Behavior/Safety Hazard/Eat & Drink/Eat & drink {video_number}.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_number}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_h = int(H * UPPER_FRACTION)

    cx_hist = []
    frame_idx = []

    bg = None
    frame_count = 0
    frames_for_save = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % PROCESS_EVERY != 0:
            continue

        roi = cv2.cvtColor(frame[:roi_h, :], cv2.COLOR_BGR2GRAY)

        # Simple background model (running average)
        if bg is None:
            bg = roi.astype("float32")
        cv2.accumulateWeighted(roi, bg, 0.01)
        diff = cv2.absdiff(roi, cv2.convertScaleAbs(bg))

        # Binary mask
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # --- Column projection ---
        proj = np.sum(mask, axis=0)  # sum vertically for each column
        if np.sum(proj) < 1e-6:
            continue  # no object

        # Centroid = weighted average of columns
        cols = np.arange(W)
        cx = np.sum(proj * cols) / np.sum(proj)
        cx_hist.append(cx)
        frame_idx.append(frame_count)

        # Draw visualization
        vis = frame.copy()
        cv2.line(vis, (int(cx), 0), (int(cx), roi_h), (0, 255, 255), 2)
        cv2.rectangle(vis, (0, 0), (W, roi_h), (255, 255, 255), 2)
        frames_for_save.append(vis)

    cap.release()

    if len(cx_hist) < 5:
        print("Not enough frames detected.")
        return

    cx_hist = np.array(cx_hist)
    disp = cx_hist - cx_hist[0]
    amp = np.max(disp) - np.min(disp)
    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO * W)

    dx = np.diff(disp)
    dx_sign = np.sign(dx)
    sig_mask = np.abs(dx) > max(2.0, 0.003 * W)
    dir_changes = np.sum((dx_sign[1:] * dx_sign[:-1] < 0) & sig_mask[1:])

    oscillatory = (amp >= amp_thresh) and (dir_changes >= MIN_DIR_CHANGES)
    print(f"Oscillation verdict: {'YES' if oscillatory else 'NO'} | Amp: {amp:.2f}px | Dir changes: {dir_changes}")

    # Save annotated frames
    out_root = "/data1/yasir/Data/Safety Hazard/Hanging Material" if oscillatory else "/data1/yasir/Data/False Positive/"
    output_folder_path = os.path.join(out_root, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_folder_path, exist_ok=True)

    for i, f in enumerate(frames_for_save):
        cv2.imwrite(os.path.join(output_folder_path, f"frame_{i}.jpg"), f)


if __name__ == "__main__":
    for video_number in range(1, 7):
        main(video_number)
