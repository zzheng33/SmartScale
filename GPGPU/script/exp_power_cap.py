import os
import subprocess
import time
import signal
import argparse
import csv

num_gpu = 4

# Define paths and executables
home_dir = os.path.expanduser('~')
script_dir = os.path.dirname(os.path.abspath(__file__))
python_executable = subprocess.getoutput('which python3')  # Adjust based on your Python version

# scripts for CPU, GPU power monitoring
read_cpu_power = os.path.join(script_dir, "power_util/read_cpu_power.py")
read_gpu_power = os.path.join(script_dir, "power_util/read_gpu_power.py")
read_gpu_metrics = os.path.join(script_dir, "power_util/read_gpu_metrics.py")
read_cpu_ips = os.path.join(script_dir, "power_util/read_cpu_ips.py")
read_mem = os.path.join(script_dir, "power_util/read_mem.py")
read_cpu_metrics = os.path.join(script_dir, "power_util/read_cpu_metrics.py")

# scritps for running various benchmarks
run_app = os.path.join(script_dir, "run_benchmark/run_app.py")

ecp_benchmarks = ['XSBench','miniGAN','CRADL','sw4lite','Laghos','bert_large','UNet','Resnet50','lammps','gromacs',"NAMD"]
npb_benchmarks = ['bt','cg','ep','ft','is','lu','mg','sp','ua','miniFE','LULESH','Nekbone']
hec_benchmarks = ["addBiasResidualLayerNorm", "aobench", "background-subtract", "chacha20", "convolution3D", "dropout", "extrema", "fft", "kalman", "knn", "softmax", "stencil3d", "zmddft", "zoom"]
altis_benchmarks_0 = ["maxflops"]
altis_benchmarks_1 = ['bfs','gemm','gups','pathfinder','sort']
altis_benchmarks_2 = ['cfd','cfd_double','fdtd2d','kmeans','lavamd',
                      'nw','particlefilter_float','particlefilter_naive','raytracing',
                      'srad','where']


spec_benchmarks = ['lbm', 'cloverleaf', 'tealeaf', 'minisweep', 'pot3d', 'miniweather', 'hpgmg']
spec_benchmarks = ['miniweather','hpgmg']
cpu_caps = [700]
# GPU_ct = [1,2,3,4]
GPU_ct = [1]
gpu_caps = [700]






def run_benchmark(benchmark_script_dir,benchmark, suite, test, size,cap_type):

    def cap_exp(g_cnt, cpu_cap, gpu_cap, output_cpu_power, output_gpu_power,output_ips, output_gpu_metrics,output_mem, output_cpu_metrics, output_runtime):

        # gpu_cap = min(gpu_cap / g_cnt, 700)

        subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(cpu_cap), str(gpu_cap)], check=True)
        time.sleep(0.2)  # Wait for the power caps to take effect

        # Run the benchmark
        start = time.time()

        run_benchmark_command = f"bash {os.path.join(home_dir, benchmark_script_dir, f'{benchmark}.sh')} {g_cnt}"

        benchmark_process = subprocess.Popen(run_benchmark_command, shell=True)
        benchmark_pid = benchmark_process.pid


        # # monitor GPU metrics
        monitor_command_gpu_metrics = f"{python_executable} {read_gpu_metrics}  --output_csv {output_gpu_metrics} --pid {benchmark_pid} --num_gpu {num_gpu}"
        monitor_process4 = subprocess.Popen(monitor_command_gpu_metrics, shell=True, stdin=subprocess.PIPE,universal_newlines=True)

        benchmark_process.wait()  # Wait for the benchmark to complete

        end = time.time()
        elapsed_time = end - start

        # Write runtime to CSV (append mode)
        file_exists = os.path.exists(output_runtime)
        with open(output_runtime, 'a') as f:
            if not file_exists:
                f.write(f"power_cap,gpu_count,runtime_seconds\n")
            f.write(f"{gpu_cap},{g_cnt},{elapsed_time}\n")

        
################## end helper function ####################
    
    cpu_cap = 700

    # Create output directory and runtime file
    output_dir = f"../data/H100/{suite}_power_motif/{benchmark}"
    os.makedirs(output_dir, exist_ok=True)
    output_runtime = f"{output_dir}/runtime.csv"

    # Delete existing runtime file to start fresh
    if os.path.exists(output_runtime):
        os.remove(output_runtime)

    for g_cnt in GPU_ct:
        for gpu_cap in gpu_caps:
            output_cpu_power = f"{output_dir}/{gpu_cap}_cpu_power.csv"
            output_gpu_power = f"{output_dir}/{gpu_cap}_{g_cnt}_gpu_power.csv"
            output_ips = f"{output_dir}/{gpu_cap}_{g_cnt}_ips.csv"
            output_mem = f"{output_dir}/{gpu_cap}_{g_cnt}_mem.csv"
            output_gpu_metrics = f"{output_dir}/{gpu_cap}_{g_cnt}_gpu_metrics.csv"
            output_cpu_metrics = f"{output_dir}/{gpu_cap}_{g_cnt}_cpu_metrics.csv"
            cap_exp(g_cnt, cpu_cap, gpu_cap, output_cpu_power, output_gpu_power,output_ips,output_gpu_metrics,output_mem,output_cpu_metrics,output_runtime)



    subprocess.run([os.path.join(script_dir, "power_util/cap.sh"), str(700), str(700)], check=True)


if __name__ == "__main__":

   # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run benchmarks and monitor power consumption.')
    parser.add_argument('--benchmark', type=str, help='Optional name of the benchmark to run', default=None)
    parser.add_argument('--test', type=int, help='whether it is a test run', default=None)
    parser.add_argument('--suite', type=int, help='0 for ECP, 1 for ALTIS, 2 for npb+ecp', default=1)
    parser.add_argument('--benchmark_size', type=int, help='0 for big, 1 for small', default=0)
    parser.add_argument('--cap_type', type=int, help='0 for cpu, 1 for gpu, 2 for dual', default=2)
    parser.add_argument('--num_gpu', type=int, default=1)

    args = parser.parse_args()
    benchmark = args.benchmark
    test = args.test
    suite = args.suite
    benchmark_size = args.benchmark_size
    cap_type = args.cap_type
    # num_gpu = args.num_gpu


    if suite == 0 or suite ==5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/ecp_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in ecp_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"ecp",test,benchmark_size,cap_type)
    

    if suite == 1 or suite ==5:
        # Map of benchmarks to their paths
        benchmark_paths = {
            "level0": altis_benchmarks_0,
            "level1": altis_benchmarks_1,
            "level2": altis_benchmarks_2
        }
    
        if benchmark:
            # Find which level the input benchmark belongs to
            found = False
            for level, benchmarks in benchmark_paths.items():
                if benchmark in benchmarks:
                    benchmark_script_dir = f"power/GPGPU/script/run_benchmark/altis_script/{level}"
                    run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
                    found = True
                    break
        else:
    
            for benchmark in altis_benchmarks_0:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level0"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_1:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level1"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)
            
            
            for benchmark in altis_benchmarks_2:
                if benchmark_size==0:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                else:
                    benchmark_script_dir = "power/GPGPU/script/run_benchmark/altis_script/level2"
                run_benchmark(benchmark_script_dir, benchmark,"altis",test,benchmark_size,cap_type)

    if suite == 2 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/npb_script/big/"
        if benchmark_size==1:
            benchmark_script_dir = f"power/GPGPU/script/run_benchmark/npb_script/small/"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"npb",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in npb_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"npb",test,benchmark_size,cap_type)


    if suite == 3 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/hec_script"
         # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)
        # run all ecp benchmarks
        else:
            for benchmark in hec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"hec",test,benchmark_size,cap_type)

    if suite == 4 or suite == 5:
        benchmark_script_dir = f"power/GPGPU/script/run_benchmark/spec_script"
        # single test
        if benchmark:
            run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)
        # run all spec benchmarks
        else:
            for benchmark in spec_benchmarks:
                run_benchmark(benchmark_script_dir, benchmark,"spec",test,benchmark_size,cap_type)