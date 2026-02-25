# import time
# import os
# import csv
# import argparse
# import psutil
# import subprocess  


# # Function to run shell commands and return output
# def run_command(command):
#     result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#     return result.stdout.strip()


# # # Function to get real-time SM clock speed (MHz)
# # def get_sm_clock():
# #     output = run_command("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits")
# #     return float(output) * 1e6  # Convert MHz to Hz


# # Function to get DCGM metrics: fp32_active, fp64_active, fp16_active, sm_active
# def safe_float(val, default=0.0):
#     """Convert val to float, return default if it fails."""
#     try:
#         return float(val)
#     except (ValueError, TypeError):
#         return default

# def get_dcgm_metrics():
#     output = run_command("dcgmi dmon -e 1008,1007,1006,1002,100,155,1005 -d 300 -c 3")
#     lines = output.split("\n")

#     # defaults
#     fp16_active = fp32_active = fp64_active = sm_active = sm_clock = power = dram_active = 0.0

#     for line in lines:
#         values = line.split()
#         if len(values) < 9:
#             continue

#         fp16_active = safe_float(values[2])
#         fp32_active = safe_float(values[3])
#         fp64_active = safe_float(values[4])
#         sm_active   = safe_float(values[5])
#         sm_clock    = safe_float(values[6])
#         power       = safe_float(values[7])
#         dram_active = safe_float(values[8])

#         # only return if we got something useful
#         if (fp32_active > 0 or fp64_active > 0 or fp16_active > 0) and sm_active > 0:
#             return fp16_active, fp32_active, fp64_active, sm_active, sm_clock, power, dram_active

#     # fallback return if no valid line
#     return fp16_active, fp32_active, fp64_active, sm_active, sm_clock, power, dram_active


# # Function to calculate real-time FLOPS
# def calculate_flops(sm_clock_hz, fp_active, sm_active, precision="FP32"):
#     if sm_clock_hz is None or fp_active is None or sm_active is None:
#         return None

#     if precision == "FP32":
#         factor = FP32_INSTRUCTIONS_PER_CYCLE
#     elif precision == "FP64":
#         factor = 1  # FP64 has lower throughput
#     elif precision == "FP16":
#         factor = FP16_INSTRUCTIONS_PER_CYCLE
#     else:
#         return None

#     flops = sm_clock_hz * fp_active * sm_active * NUM_SMS * CUDA_CORES_PER_SM * factor
#     return flops / 1e12  # Convert to TFLOPS

# # Function to monitor GPU performance
# def monitor_gpu_performance(benchmark_pid, output_csv, interval=0.1):
#     start_time = time.time()
#     performance_data = []
    
#     while psutil.pid_exists(benchmark_pid):
#         time.sleep(interval)
#         elapsed_time = time.time() - start_time
#         # sm_clock_hz = get_sm_clock()
#         fp16_active, fp32_active, fp64_active, sm_active, sm_clock_hz, power, dram_active = get_dcgm_metrics()

#         # fp32_flops = calculate_flops(sm_clock_hz, fp32_active, sm_active, precision="FP32")
#         # fp64_flops = calculate_flops(sm_clock_hz, fp64_active, sm_active, precision="FP64")
#         # fp16_flops = calculate_flops(sm_clock_hz, fp16_active, sm_active, precision="FP16")
        
#         row = [elapsed_time, sm_clock_hz, dram_active, fp16_active, fp32_active, fp64_active, int(power)]
#         performance_data.append(row)
        
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     with open(output_csv, 'w', newline='') as file:
#         writer = csv.writer(file)
#         headers = ['Time (s)', 'SM Clock (MHz)', 'DRAM Active', 'FP16 Active', 'FP32 Active', 'FP64 Active', 'Power (W)']
#         writer.writerow(headers)
#         writer.writerows(performance_data)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Monitor GPU performance.')
#     parser.add_argument('--pid', type=int, help='PID of the benchmark process', required=True)
#     parser.add_argument('--output_csv', type=str, help='Output CSV file path', required=True)
#     parser.add_argument('--num_gpu', type=int)
#     args = parser.parse_args()
    
#     monitor_gpu_performance(args.pid, args.output_csv)





# below for multi-gpu reading

import time
import os
import csv
import argparse
import psutil
import subprocess  

num_gpu = 4

# Function to run shell commands and return output
def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return result.stdout.strip()


# # Function to get real-time SM clock speed (MHz)
# def get_sm_clock():
#     output = run_command("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits")
#     return float(output) * 1e6  # Convert MHz to Hz


# Function to get DCGM metrics for all GPUs
# Returns a list of metrics per GPU: [(gpu0_metrics), (gpu1_metrics), ...]
def get_dcgm_metrics():
    output = run_command("dcgmi dmon -e 1008,1007,1006,1002,100,155,1005 -d 300 -c 3")
    lines = output.split("\n")

    # Parse GPU data lines - collect all readings for each GPU
    # gpu_readings[gpu_id] = [reading1, reading2, reading3, ...]
    gpu_readings = {}
    for line in lines:
        values = line.strip().split()

        # Skip if not a data line or too short
        if len(values) < 9:
            continue
        if values[0] != "GPU":
            continue

        try:
            gpu_id = int(values[1])
        except ValueError:
            continue

        if gpu_id < num_gpu:
            if gpu_id not in gpu_readings:
                gpu_readings[gpu_id] = []
            gpu_readings[gpu_id].append(values[2:9])

    # For each GPU, find the first non-N/A reading for each metric
    gpu_data = {}
    for gpu_id in range(num_gpu):
        if gpu_id not in gpu_readings or len(gpu_readings[gpu_id]) == 0:
            gpu_data[gpu_id] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue

        # Initialize with N/A
        metrics = ['N/A'] * 7

        # For each metric position (0-6), find first non-N/A value across all readings
        for reading in gpu_readings[gpu_id]:
            for i in range(7):
                if metrics[i] == 'N/A' and reading[i] != 'N/A':
                    metrics[i] = reading[i]

        # Convert to floats, using 0.0 for any remaining N/A
        fp16_active = 0.0 if metrics[0] == 'N/A' else float(metrics[0])
        fp32_active = 0.0 if metrics[1] == 'N/A' else float(metrics[1])
        fp64_active = 0.0 if metrics[2] == 'N/A' else float(metrics[2])
        sm_active = 0.0 if metrics[3] == 'N/A' else float(metrics[3])
        sm_clock = 0.0 if metrics[4] == 'N/A' else float(metrics[4])
        power = 0.0 if metrics[5] == 'N/A' else float(metrics[5])
        dram_active = 0.0 if metrics[6] == 'N/A' else float(metrics[6])

        gpu_data[gpu_id] = (fp16_active, fp32_active, fp64_active, sm_active, sm_clock, power, dram_active)

    # Return data for all GPUs in order (GPU 0, 1, 2, ...)
    result = []
    for i in range(num_gpu):
        if i in gpu_data:
            result.append(gpu_data[i])
        else:
            # Default values if GPU data not found
            result.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return result





# Function to monitor GPU performance
def monitor_gpu_performance(benchmark_pid, output_csv, interval=0.2):
    start_time = time.time()
    performance_data = []

    while psutil.pid_exists(benchmark_pid):
        time.sleep(interval)
        elapsed_time = time.time() - start_time

        # Get metrics for all GPUs
        gpu_metrics = get_dcgm_metrics()

        # Build row: [time, gpu0_metrics..., gpu1_metrics..., ...]
        row = [elapsed_time]
        for fp16, fp32, fp64, sm, clock, pwr, dram in gpu_metrics:
            row.extend([clock, dram, fp16, fp32, fp64, int(pwr)])

        performance_data.append(row)

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        # Generate headers: Time, GPU0_*, GPU1_*, ...
        headers = ['Time (s)']
        for i in range(num_gpu):
            headers.extend([
                f'GPU{i}_SM_Clock (MHz)',
                f'GPU{i}_DRAM_Active',
                f'GPU{i}_FP16_Active',
                f'GPU{i}_FP32_Active',
                f'GPU{i}_FP64_Active',
                f'GPU{i}_Power (W)'
            ])

        writer.writerow(headers)
        writer.writerows(performance_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU performance.')
    parser.add_argument('--pid', type=int, help='PID of the benchmark process', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV file path', required=True)
    parser.add_argument('--num_gpu', type=int, required=True)
    args = parser.parse_args()
    num_gpu = args.num_gpu
    
    monitor_gpu_performance(args.pid, args.output_csv)