import torch
import time

def benchmark_matmul(size, device):
    print(f"\nTesting matmul with size {size}x{size} on {device} ...")

    # Create random matrices directly on device (use float32 for best GPU throughput)
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed runs
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg = total / iters

    # FLOPS estimate: matrix multiply ≈ 2 * N^3 floating-point ops
    flops = 2 * (size ** 3) / avg
    tflops = flops / 1e12

    print(f"Average time per run: {avg:.6f} s")
    print(f"Approx compute: {tflops:.2f} TFLOPS")

def main():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (from torch):", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional: enable cudnn autotuner for possible speedups
    torch.backends.cudnn.benchmark = True

    # Test different sizes (stop on OOM)
    for size in [1024, 2048, 3072]:
        try:
            benchmark_matmul(size, device)
        except RuntimeError as e:
            print(f"RuntimeError for size {size}: {e}")
            # If OOM or other runtime error, continue to next (or break if you prefer)
            continue

if __name__ == "__main__":
    main()
