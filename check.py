import time
import torch
from ultralytics import YOLO


def main():
    # ---- Basic CUDA / Torch info ----
    print("torch:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch..is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device name:", torch.cuda.get_device_name(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Load YOLOv8n and move core model to GPU ----
    model = YOLO("yolov8n.pt")          # will download once if not present
    model.to(device)
    model.model.eval()                  # eval mode for speed
    torch.backends.cudnn.benchmark = True

    # Check where the model lives
    p = next(model.model.parameters())
    print("First param device:", p.device, "dtype:", p.dtype)

    # ---- Create dummy input directly on device ----
    img_size = 640
    dummy = torch.rand(1, 3, img_size, img_size, device=device)

    # ---- Warmup ----
    print("\nWarming up...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model.model(dummy)

    # ---- Timed runs (RAW MODEL ONLY: no pre/post/NMS) ----
    num_iters = 100
    print(f"\nRunning {num_iters} raw inferences at {img_size}x{img_size}...")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.inference_mode():
        for _ in range(num_iters):
            _ = model.model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    total = t1 - t0
    fps = num_iters / total
    print(f"\nTotal time: {total:.3f} s for {num_iters} frames")
    print(f"Average FPS (raw model): {fps:.1f}")


if __name__ == "__main__":
    main()
cuda