#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time


def run_cmd(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def set_max_freq_mhz(gpu_idx, freq_mhz):
    freq_hz = int(freq_mhz) * 1_000_000
    run_cmd(
        [
            "geopmwrite",
            "NVML::GPU_CORE_FREQUENCY_MAX_CONTROL",
            "gpu",
            str(gpu_idx),
            str(freq_hz),
        ]
    )


def read_freq_status_mhz(gpu_idx):
    out = run_cmd(["geopmread", "NVML::GPU_CORE_FREQUENCY_STATUS", "gpu", str(gpu_idx)])
    return float(out) / 1_000_000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple single-GPU PyTorch stress benchmark (manual power monitoring)."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Benchmark duration in seconds")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square GEMM matrix size")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Tensor dtype for GEMM",
    )
    parser.add_argument(
        "--max-freq-mhz",
        type=int,
        default=None,
        help="Optional: set GPU max core frequency cap (MHz) before running",
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=5.0,
        help="Progress print interval in seconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --gpu {args.gpu}. Available CUDA devices: {torch.cuda.device_count()}."
        )

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.max_freq_mhz is not None:
        set_max_freq_mhz(args.gpu, args.max_freq_mhz)
        print(f"Set GPU {args.gpu} max freq cap to {args.max_freq_mhz} MHz")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    n = args.matrix_size
    print(
        f"Starting stress: gpu={args.gpu}, duration={args.duration_s}s, "
        f"matrix={n}x{n}, dtype={args.dtype}"
    )
    print("Monitor power in another terminal, e.g.:")
    print(f"  watch -n 0.5 nvidia-smi --id={args.gpu} --query-gpu=power.draw,clocks.sm,utilization.gpu --format=csv,noheader,nounits")

    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)
    c = torch.empty((n, n), device=device, dtype=dtype)

    start = time.time()
    next_log = start + args.log_interval_s
    iters = 0

    with torch.no_grad():
        while True:
            # Keep inputs fixed to avoid fp16/bf16 value blow-up over iterations.
            torch.matmul(a, b, out=c)
            iters += 1

            if iters % 3 == 0:
                torch.cuda.synchronize(device)
                now = time.time()
                if now >= start + args.duration_s:
                    break
                if now >= next_log:
                    freq_mhz = read_freq_status_mhz(args.gpu)
                    elapsed = now - start
                    print(f"elapsed={elapsed:6.1f}s  iterations={iters:8d}  freq_status={freq_mhz:.1f} MHz")
                    next_log = now + args.log_interval_s

    torch.cuda.synchronize(device)
    total = time.time() - start
    print(f"Done. Total time: {total:.2f}s, iterations: {iters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
