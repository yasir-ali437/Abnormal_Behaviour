import os
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

# ================ CONFIG (same spirit as yours) =================
OUTPUT_DIR = "./hanging_outputs"

UPPER_FRACTION   = 0.3     # analyze top 90% of frame
PROCESS_EVERY    = 2        # skip frames for speed

# Detection filters
MIN_ASPECT_RATIO = 1.2      # tall shape filter h/w
MIN_HEIGHT_RATIO = 0.50     # object height ≥ 50% of frame
MIN_AREA_RATIO   = 0.0005   # relative area threshold
MAX_WIDTH_RATIO  = 0.40     # reject extremely wide objects

# Oscillation logic
AMP_MIN_PIXELS   = 8.0
AMP_MIN_RATIO    = 0.01
MIN_DIR_CHANGES  = 6
MIN_FLOW_FLIPS   = 8

# GPU background model
EMA_ALPHA        = 0.02     # background EMA update
FG_ABS_THRESH    = 0.08     # 0..1 grayscale difference threshold

# Morphology
OPEN_K           = 3        # opening kernel size (odd)
CLOSE_K          = 5        # closing kernel size (odd)
# ================================================================


def to_gray_gpu(frame_bgr_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert BGR uint8 HxWx3 to grayscale GPU tensor in [0,1], shape [1,1,H,W].
    """
    # BGR to grayscale using standard coefficients
    # gray = 0.114*B + 0.587*G + 0.299*R
    b = torch.from_numpy(frame_bgr_np[..., 0]).to(device=device, dtype=torch.float32)
    g = torch.from_numpy(frame_bgr_np[..., 1]).to(device=device, dtype=torch.float32)
    r = torch.from_numpy(frame_bgr_np[..., 2]).to(device=device, dtype=torch.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    return gray.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def pad_same_2d(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Pad for 'same' pooling with odd kernel k.
    """
    p = k // 2
    return F.pad(x, (p, p, p, p), mode="replicate")


def erode_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    """
    Binary erosion using min-pool via max-pool on the inverted values.
    mask01: float tensor in {0,1}, shape [1,1,H,W]
    """
    # Erosion = minpool(mask), we can simulate via maxpool of inverted
    x = pad_same_2d(1.0 - mask01, k)
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return 1.0 - y


def dilate_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    """
    Binary dilation via max-pool.
    """
    x = pad_same_2d(mask01, k)
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return y


def open_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    return dilate_gpu(erode_gpu(mask01, k), k)


def close_gpu(mask01: torch.Tensor, k: int) -> torch.Tensor:
    return erode_gpu(dilate_gpu(mask01, k), k)


@torch.no_grad()
def phase_corr_dx(prev_crop: torch.Tensor, curr_crop: torch.Tensor) -> float:
    """
    Estimate horizontal translation (dx) between prev and curr crops using phase correlation on GPU.
    Inputs: [1,1,h,w] float tensors (0..1), same size, device cuda.
    Returns: dx (pixels). Positive = shift to the right in current vs prev.
    """
    # Taper with a Hann window to reduce edge effects
    _, _, h, w = curr_crop.shape
    wy = torch.hann_window(h, device=curr_crop.device, dtype=curr_crop.dtype).view(1,1,h,1)
    wx = torch.hann_window(w, device=curr_crop.device, dtype=curr_crop.dtype).view(1,1,1,w)
    w2d = wy * wx

    A = prev_crop * w2d
    B = curr_crop * w2d

    # FFT
    FA = torch.fft.rfft2(A)
    FB = torch.fft.rfft2(B)

    R = FA * torch.conj(FB)
    denom = torch.abs(R)
    R = torch.where(denom > 1e-8, R / denom, torch.zeros_like(R))

    r = torch.fft.irfft2(R, s=(h, w))  # correlation map

    # Argmax
    idx = torch.argmax(r)
    peak_y = (idx // w).item()
    peak_x = (idx % w).item()

    # Wrap to negative if peak beyond half
    if peak_x > w // 2:
        peak_x = peak_x - w
    if peak_y > h // 2:
        peak_y = peak_y - h

    return float(peak_x)


def choose_tall_object(contours, W, H, roi_y0):
    """
    Select the best tall object box based on area * aspect.
    Returns: (chosen_box(x,y,w,h), chosen_cx, chosen_cy, best_score) or (None,None,None,None)
    """
    best_score = -1.0
    chosen = (None, None, None, None)

    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        X, Y = x, y  # within ROI coords; we'll add roi_y0 later for global

        aspect = h / (w + 1e-6)
        if aspect < MIN_ASPECT_RATIO or h < H * MIN_HEIGHT_RATIO or w > W * MAX_WIDTH_RATIO:
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

            chosen = ((X, Y + roi_y0, w, h), cx_rel, cy_rel + roi_y0, best_score)

    return chosen


def main(video_number):
    assert torch.cuda.is_available(), "CUDA GPU not available. Please install CUDA-enabled PyTorch."
    device = torch.device("cuda")

    video_path = f"/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/{video_number} hanging material.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_y0, roi_h = 0, int(H * UPPER_FRACTION)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_video = os.path.join(OUTPUT_DIR, "hanging_annotated.mp4")
    # If you want an annotated video, uncomment below:
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # vw = cv2.VideoWriter(out_video, fourcc, FPS, (W, H))

    # Histories
    cx_hist, frame_idx = [], []
    flow_dx_hist = []
    box_hist = []
    frame_list = []

    min_area = max(200, int(MIN_AREA_RATIO * W * H))

    bg_roi = None              # [1,1,roi_h,W] on GPU
    prev_gray_roi = None       # [1,1,roi_h,W] on GPU
    prev_obj_crop = None       # for phase correlation on object crop

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % PROCESS_EVERY != 0:
            continue

        # ---- prep on GPU ----
        gray = to_gray_gpu(frame, device)           # [1,1,H,W]
        gray_roi = gray[:, :, roi_y0:roi_y0 + roi_h, 0:W]  # [1,1,roi_h,W]

        # ---- background init/update (EMA) ----
        if bg_roi is None:
            bg_roi = gray_roi.clone()
        # Foreground mask from absolute difference
        diff = torch.abs(gray_roi - bg_roi)  # [1,1,roi_h,W]
        fg_mask = (diff > FG_ABS_THRESH).float()   # binary 0/1 on GPU

        # Basic morphology: open then close to remove noise / fill gaps
        fg_mask = open_gpu(fg_mask, OPEN_K)
        fg_mask = close_gpu(fg_mask, CLOSE_K)
        # Binarize hard
        fg_mask = (fg_mask > 0.5).float()

        # Update EMA background (simple; full-frame)
        bg_roi = (1.0 - EMA_ALPHA) * bg_roi + EMA_ALPHA * gray_roi

        # ---- contours (CPU) ----
        # convert to uint8 mask for cv2
        fg_np = (fg_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        # Remove tiny blobs fast with OpenCV morphology (optional, CPU)
        if min_area > 0:
            # A small opening/closing on CPU to clean stray pixels (cheap)
            kcpu = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            fg_np = cv2.morphologyEx(fg_np, cv2.MORPH_OPEN, kcpu)

        contours, _ = cv2.findContours(fg_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        chosen_box, chosen_cx, chosen_cy, best_score = choose_tall_object(contours, W, H, roi_y0)

        # ---- horizontal motion sign (phase correlation on GPU) ----
        dx_est = 0.0
        if chosen_box is not None:
            x, y, w, h = chosen_box

            # clamp to ROI to avoid borders
            y0 = max(roi_y0, y)
            y1 = min(roi_y0 + roi_h, y + h)
            x0 = max(0, x)
            x1 = min(W, x + w)

            if y1 > y0 and x1 > x0:
                # crop current ROI tensor to the object
                # gray[:, :, :] is [1,1,H,W]; take object slice
                curr_obj = gray[:, :, y0:y1, x0:x1]

                if prev_obj_crop is not None and \
                   prev_obj_crop.shape[-2] == curr_obj.shape[-2] and \
                   prev_obj_crop.shape[-1] == curr_obj.shape[-1]:
                    # Estimate dx via phase correlation
                    try:
                        dx_est = phase_corr_dx(prev_obj_crop, curr_obj)
                    except Exception:
                        dx_est = 0.0

                prev_obj_crop = curr_obj.clone()

        # ---- record histories ----
        if chosen_box is not None:
            cx_hist.append(chosen_cx)
            frame_idx.append(frame_count)
            flow_dx_hist.append(dx_est)
            box_hist.append(chosen_box)

        # ---- draw annotation (CPU) ----
        vis = frame.copy()
        cv2.rectangle(vis, (0, roi_y0), (W, roi_y0 + roi_h), (255, 255, 255), 2)
        if chosen_box is not None:
            x, y, w, h = chosen_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (int(chosen_cx), int(chosen_cy)), 4, (0, 0, 255), -1)
            # Flow arrow from dx_est sign
            arrow_dx = int(15 * (1 if dx_est > 0 else (-1 if dx_est < 0 else 0)))
            cv2.arrowedLine(vis, (int(chosen_cx), int(chosen_cy)),
                            (int(chosen_cx + arrow_dx), int(chosen_cy)),
                            (0, 255, 255), 2, tipLength=0.4)

        # If writing a video:
        # vw.write(vis)
        frame_list.append(vis)

        prev_gray_roi = gray_roi  # keep if needed elsewhere

    cap.release()
    # if 'vw' in locals(): vw.release()

    # ======= Oscillation Decision =======
    if len(cx_hist) < 5:
        print("Not enough frames detected.")
        return

    cx = np.array(cx_hist, dtype=float)
    disp = cx - cx[0]
    amp = np.nanmax(disp) - np.nanmin(disp)
    amp_thresh = max(AMP_MIN_PIXELS, AMP_MIN_RATIO * W)

    dx = np.diff(disp)
    dx_mag = np.abs(dx)
    dx_sign = np.sign(dx)
    sig_mask = dx_mag > max(2.0, 0.003 * W)
    dir_changes = int(np.sum((dx_sign[1:] * dx_sign[:-1] < 0) & sig_mask[1:]))

    # From phase correlation signs
    if len(flow_dx_hist) >= 2:
        fsign = np.sign(np.array(flow_dx_hist, dtype=float))
        flow_changes = int(np.sum(fsign[1:] * fsign[:-1] < 0))
    else:
        flow_changes = 0

    oscillatory = (amp >= amp_thresh) and (dir_changes >= MIN_DIR_CHANGES or flow_changes >= MIN_FLOW_FLIPS)
    print(f"Oscillation verdict: {'YES' if oscillatory else 'NO'}")
    print(f"Δx amplitude: {amp:.2f}px (thr {amp_thresh:.2f}) | dir_changes: {dir_changes} | flow_flips: {flow_changes}")

    # ======= Save frame outputs to the requested folder tree =======
    if oscillatory:
        out_root = "/data1/yasir/Data/Safety Hazard/Hanging Material"
    else:
        out_root = "/data1/yasir/Data/False Positive/"
    output_folder_path = os.path.join(out_root, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_folder_path, exist_ok=True)

    for i, f in enumerate(frame_list):
        filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
        cv2.imwrite(filename, f)


if __name__ == "__main__":
    # Example: 1..32 videos named like "1 hanging material.mp4"
    for video_number in range(1, 33):
        main(video_number)
