import os
import time
import csv
import psutil
import subprocess
import argparse

# Function to read the socket power using e_smi_tool
def read_socket_power():
    try:
        result = subprocess.run(['sudo', '/home/cc/esmi_ib_library/build/e_smi_tool', '--showsockpower'], capture_output=True, text=True)
        output = result.stdout

        # The rest of the parsing logic
        for line in output.splitlines():
            if "Power (Watts)" in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    power_socket_0 = float(parts[2].strip())
                    power_socket_1 = float(parts[3].strip())
                    return [power_socket_0, power_socket_1]
                else:
                    raise ValueError("Unexpected format in power line: " + line)

        raise ValueError("Failed to parse power values from e_smi_tool output.")

    except subprocess.CalledProcessError as e:
        print(f"Error executing e_smi_tool: {e}")
        return [0, 0]
    except Exception as e:
        print(f"Error reading socket power: {e}")
        return [0, 0]


# Function to monitor power consumption updated to add socket powers together
def monitor_power(benchmark_pid, output_csv, avg, interval=0.2):
    start_time = time.time()
    power_data = []
    total_energy = 0
    
    while psutil.pid_exists(benchmark_pid):
        time.sleep(interval)
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        power_values = read_socket_power()
        total_cpu_power = sum(power_values)
        total_energy += total_cpu_power * interval
        
        power_data.append([elapsed_time, total_cpu_power])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if avg:
        with open(output_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            if os.stat(output_csv).st_size == 0:  # If the file is empty, add the header
                writer.writerow(['CPU_E (J)'])
            writer.writerow([round(total_energy, 2)])  # Append the total energy
    else:
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time (s)', 'Package Power (W)'])
            writer.writerows(power_data)

# Main function and argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor total power usage using e_smi_tool for all CPU sockets.')
    parser.add_argument('--pid', type=int, help='PID of the benchmark process', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV file path', required=True)
    parser.add_argument('--avg', type=str, help='avg_power', default=0)
    args = parser.parse_args()

    monitor_power(args.pid, args.output_csv, args.avg)





