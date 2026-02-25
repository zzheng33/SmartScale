#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import time


def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def geopm_read(signal, gpu):
    return float(run_cmd(["geopmread", signal, "gpu", str(gpu)]))


def geopm_write(control, gpu, value):
    run_cmd(["geopmwrite", control, "gpu", str(gpu), str(value)])


def mhz_to_hz(mhz):
    return int(mhz * 1_000_000)


def hz_to_mhz(hz):
    return hz / 1_000_000.0


def measure_one_freq(gpu, freq_mhz, warmup_s, measure_s, sample_interval_s, matrix_size):
    import torch

    geopm_write("NVML::GPU_CORE_FREQUENCY_MAX_CONTROL", gpu, mhz_to_hz(freq_mhz))
    time.sleep(0.3)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)

    # Warmup
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        c = torch.matmul(a, b)
        a, b = b, c
    torch.cuda.synchronize(device)

    # Measure
    power = []
    freq = []
    t1 = time.time()
    next_sample = t1
    while time.time() - t1 < measure_s:
        c = torch.matmul(a, b)
        a, b = b, c
        torch.cuda.synchronize(device)
        now = time.time()
        if now >= next_sample:
            power.append(geopm_read("NVML::GPU_POWER", gpu))
            freq.append(hz_to_mhz(geopm_read("NVML::GPU_CORE_FREQUENCY_STATUS", gpu)))
            next_sample = now + sample_interval_s

    if not power:
        raise RuntimeError("No samples collected. Increase measure time.")

    return {
        "set_freq_mhz": int(freq_mhz),
        "max_power_w": max(power),
        "avg_power_w": sum(power) / len(power),
        "max_obs_freq_mhz": max(freq),
        "n_samples": len(power),
    }


def build_mapping(rows, pmin=100, pmax=200, pstep=10):
    points = []
    for p in range(pmin, pmax + 1, pstep):
        safe = [r for r in rows if r["max_power_w"] <= p]
        if safe:
            f = max(safe, key=lambda x: x["set_freq_mhz"])["set_freq_mhz"]
        else:
            f = min(rows, key=lambda x: x["set_freq_mhz"])["set_freq_mhz"]
        points.append((p, int(f)))

    # enforce monotonicity
    out = []
    best = -1
    for p, f in points:
        if f < best:
            f = best
        best = f
        out.append((p, f))
    return out


def print_function(mapping):
    print("\n# Copy this function")
    print(f"MAPPING_POINTS = {mapping}")
    print("def freq_for_power(power_watts):")
    print("    pts = MAPPING_POINTS")
    print("    if power_watts <= pts[0][0]:")
    print("        return int(pts[0][1])")
    print("    for (p0, f0), (p1, f1) in zip(pts, pts[1:]):")
    print("        if power_watts <= p1:")
    print("            x = (power_watts - p0) / float(p1 - p0)")
    print("            return int(round(f0 + x * (f1 - f0)))")
    print("    return int(pts[-1][1])")


def parse_args():
    ap = argparse.ArgumentParser(description="Simple GPU freq->power mapper (PyTorch + GEOPM)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--measure-s", type=float, default=10.0)
    ap.add_argument("--sample-interval-s", type=float, default=0.2)
    ap.add_argument("--matrix-size", type=int, default=8192)
    ap.add_argument("--freq-step-mhz", type=int, default=60)
    ap.add_argument("--target-min-w", type=int, default=100)
    ap.add_argument("--target-max-w", type=int, default=200)
    ap.add_argument("--target-step-w", type=int, default=10)
    ap.add_argument(
        "--out-csv",
        default="power-GPU-count/GPGPU/data/freq_power_map_simple.csv",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    min_mhz = int(round(hz_to_mhz(geopm_read("NVML::GPU_CORE_FREQUENCY_MIN_AVAIL", args.gpu))))
    max_mhz = int(round(hz_to_mhz(geopm_read("NVML::GPU_CORE_FREQUENCY_MAX_AVAIL", args.gpu))))
    step = max(1, args.freq_step_mhz)

    freqs = list(range(max_mhz, min_mhz - 1, -step))
    if freqs[-1] != min_mhz:
        freqs.append(min_mhz)

    print(f"GPU {args.gpu} freq sweep: {max_mhz} -> {min_mhz} MHz, step {step} MHz")
    print(f"points={len(freqs)}, per-point={args.warmup_s + args.measure_s:.1f}s")

    rows = []
    for i, f in enumerate(freqs, 1):
        print(f"[{i}/{len(freqs)}] testing {f} MHz...")
        r = measure_one_freq(
            gpu=args.gpu,
            freq_mhz=f,
            warmup_s=args.warmup_s,
            measure_s=args.measure_s,
            sample_interval_s=args.sample_interval_s,
            matrix_size=args.matrix_size,
        )
        rows.append(r)
        print(
            f"  max_power={r['max_power_w']:.2f}W "
            f"avg_power={r['avg_power_w']:.2f}W "
            f"max_obs_freq={r['max_obs_freq_mhz']:.1f}MHz"
        )

    mapping = build_mapping(
        rows,
        pmin=args.target_min_w,
        pmax=args.target_max_w,
        pstep=args.target_step_w,
    )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "set_freq_mhz",
                "max_power_w",
                "avg_power_w",
                "max_obs_freq_mhz",
                "n_samples",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("\nPower->MaxFreq mapping (W -> MHz):")
    for p, f in mapping:
        print(f"  {p} -> {f}")
    print(f"\nraw data csv: {args.out_csv}")
    print_function(mapping)
    return 0


if __name__ == "__main__":
    sys.exit(main())
