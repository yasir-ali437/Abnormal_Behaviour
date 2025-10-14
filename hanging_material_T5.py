import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ===================== CONFIG =====================
OUTPUT_DIR = "./hanging_outputs_gpu"

# ROI
UPPER_FRACTION   = 0.90        # analyze top 90% of frame
PROCESS_EVERY    = 2           # skip frames for speed (use 1 for max accuracy)

# Detection filters
MIN_ASPECT_RATIO = 1.20        # tall shape filter (h/w)
MIN_HEIGHT_RATIO = 0.50        # object height ≥ 50% of frame height
MIN_AREA_RATIO   = 0.0005      # relative to full frame area
MAX_WIDTH_RATIO  = 0.40        # reject extremely wide blobs

# Oscillation logic
AMP_MIN_PIXELS   = 8.0
AMP_MIN_RATIO    = 0.01
MIN_DIR_CHANGES  = 6           # direction flips of dx
SIG_DX_RATIO     = 0.003       # significant dx threshold as fraction of width
SIG_DX_PIXELS    = 2.0

# Background model (GPU)
EMA_ALPHA        = 0.02        # running average update
FG_ABS_THRESH    = 0.08        # 0..1 grayscale difference

# Morphology (GPU): keep tiny and cheap
OPEN_K           = 3           # odd
CLOSE_K          = 5           # odd

# Kalman Filter (1D on x)
KF_PROCESS_NOISE = 2.0         # q
KF_MEAS_NOISE    = 6.0         # r
# ==================================================


# ---------------- GPU helpers ----------------
def to_gray_gpu(frame_bgr_np: np.ndarray, device) -> torch.Tensor:
    """BGR uint8 HxWx3 -> [1,1,H,W] float in [0,1] on GPU."""
    b = torch.from_numpy(frame_bgr_np[..., 0]).to(device=device, dtype=torch.float32)
    g = torch.from_numpy(frame_bgr_np[..., 1]).to(device=device, dtype=torch.float32)
    r = torch.from_numpy(frame_bgr_np[..., 2]).to(device=device, dtype=torch.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    return gray.unsqueeze(0).unsqueeze(0)

def pad_same(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    return F.pad(x, (p, p, p, p), mode="replicate")

def erode_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    # erosion = minpool -> via maxpool on inverted
    x = pad_same(1.0 - mask01, k)
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return 1.0 - y

def dilate_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    x = pad_same(mask01, k)
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return y

def open_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    return dilate_gpu(erode_gpu(mask01, k), k)

def close_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    return erode_gpu(dilate_gpu(mask01, k), k)

# --------------- Kalman Filter (1D x) ---------------
class Kalman1D:
    """
    State: [x, vx]
    x_{k+1} = [1 dt; 0 1] x_k + w,     w ~ N(0,q*I)
    z_k     = [1  0] x_k + v,          v ~ N(0,r)
    """
    def __init__(self, x0: float, dt: float, q: float, r: float):
        self.dt = dt
        self.q = q
        self.r = r
        # State
        self.x = np.array([[x0], [0.0]], dtype=np.float32)
        # Covariance
        self.P = np.eye(2, dtype=np.float32) * 1e3
        # Matrices
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]], dtype=np.float32)
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)
        self.Q = np.array([[q, 0.0],
                           [0.0, q]], dtype=np.float32)
        self.R = np.array([[r]], dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        z = np.array([[z]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def current(self) -> float:
        return float(self.x[0, 0])

# ---------------- Utility ----------------
def choose_tall_object(contours, W, H, roi_y0):
    """
    Select main tall object by area*aspect; return (x,y,w,h, cx, cy) in full-frame coords or None.
    """
    best_score = -1.0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / (w + 1e-6)
        if aspect < MIN_ASPECT_RATIO:             # not tall
            continue
        if h < H * MIN_HEIGHT_RATIO:              # not tall enough
            continue
        if w > W * MAX_WIDTH_RATIO:               # too wide
            continue
        score = area * aspect
        if score > best_score:
            best_score = score
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_rel = int(M["m10"] / M["m00"])
                cy_rel = int(M["m01"] / M["m00"])
            else:
                cx_rel, cy_rel = x + w // 2, y + h // 2
            best = (x, y + roi_y0, w, h, cx_rel, cy_rel + roi_y0)
    return best  # or None

# ---------------- Main ----------------
def main(video_number: int):
    assert torch.cuda.is_available(), "CUDA GPU not available. Install CUDA-enabled PyTorch."
    device = torch.device("cuda")
    video_path = f"/data1/yasir/Data/Abnormal Behavior2/Abnormal Behavior/Safety Hazard/Eat & Drink/Eat & drink {video_number}.mp4"
    # video_path = f"/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/{video_number} hanging material.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / max(FPS, 1.0)

    roi_y0, roi_h = 0, int(H * UPPER_FRACTION)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Histories
    raw_cx_hist, kf_cx_hist, frame_idx, boxes = [], [], [], []
    frames_vis = []

    min_area = max(200, int(MIN_AREA_RATIO * W * H))

    bg_roi = None  # [1,1,roi_h,W] on GPU
    kf = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % PROCESS_EVERY != 0:
            continue

        # --------- GPU preproc ---------
        gray = to_gray_gpu(frame, device)  # [1,1,H,W]
        gray_roi = gray[:, :, roi_y0:roi_y0 + roi_h, 0:W]  # [1,1,roi_h,W]

        # Init / update background (EMA)
        if bg_roi is None:
            bg_roi = gray_roi.clone()
        diff = torch.abs(gray_roi - bg_roi)               # [1,1,roi_h,W]
        fg_mask = (diff > FG_ABS_THRESH).float()
        # light morphology
        if OPEN_K > 1:
            fg_mask = open_gpu(fg_mask, OPEN_K)
        if CLOSE_K > 1:
            fg_mask = close_gpu(fg_mask, CLOSE_K)
        fg_mask = (fg_mask > 0.5).float()

        # Background update
        bg_roi = (1.0 - EMA_ALPHA) * bg_roi + EMA_ALPHA * gray_roi

        # --------- Contours on CPU (cheap) ---------
        mask_np = (fg_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        # quick CPU cleanup for tiny speckles
        if min_area > 0:
            kcpu = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kcpu)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # size filter
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        choice = choose_tall_object(contours, W, H, roi_y0)
        if choice is None:
            # no valid object this frame; keep drawing ROI only
            vis = frame.copy()
            cv2.rectangle(vis, (0, roi_y0), (W, roi_y0 + roi_h), (255, 255, 255), 2)
            frames_vis.append(vis)
            continue

        x, y, w, h, cx, cy = choice
        raw_cx_hist.append(float(cx))
        frame_idx.append(frame_count)
        boxes.append((x, y, w, h))

        # --------- Kalman update ---------
        if kf is None:
            kf = Kalman1D(x0=float(cx), dt=dt * PROCESS_EVERY, q=KF_PROCESS_NOISE, r=KF_MEAS_NOISE)
        else:
            kf.predict()
            kf.update(float(cx))
        kf_cx = kf.current()
        kf_cx_hist.append(kf_cx)

        # --------- Annotate ---------
        vis = frame.copy()
        cv2.rectangle(vis, (0, roi_y0), (W, roi_y0 + roi_h), (255, 255, 255), 2)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)          # raw centroid
        cv2.line(vis, (int(kf_cx), 0), (int(kf_cx), roi_h), (0, 255, 255), 2)  # filtered x
        frames_vis.append(vis)

    cap.release()

    # --------------- Verdict ---------------
    if len(kf_cx_hist) < 5:
        print("Not enough frames detected.")
        return

    cx = np.array(kf_cx_hist, dtype=float)
    disp = cx - cx[0]
    amp = float(np.nanmax(disp) - np.nanmin(disp))
    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO * W)

    dx = np.diff(disp)
    sig_thr = max(SIG_DX_PIXELS, SIG_DX_RATIO * W)
    sig_mask = np.abs(dx) > sig_thr
    dx_sign = np.sign(dx)
    dir_changes = int(np.sum((dx_sign[1:] * dx_sign[:-1] < 0) & sig_mask[1:]))

    oscillatory = (amp >= amp_thresh) and (dir_changes >= MIN_DIR_CHANGES)
    print(f"[{os.path.basename(video_path)}] Oscillation: {'YES' if oscillatory else 'NO'} | "
          f"Amp={amp:.1f}px Thr={amp_thresh:.1f}px | DirChanges={dir_changes} "
          f"| FramesUsed={len(cx)} (every {PROCESS_EVERY})")

    # --------------- Save outputs ---------------
    out_root = "/data1/yasir/Data/Safety Hazard/Hanging Material" if oscillatory else "/data1/yasir/Data/False Positive/"
    output_folder_path = os.path.join(out_root, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_folder_path, exist_ok=True)

    for i, f in enumerate(frames_vis):
        cv2.imwrite(os.path.join(output_folder_path, f"frame_{i}.jpg"), f)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for video_number in range(1, 7):
        main(video_number)
