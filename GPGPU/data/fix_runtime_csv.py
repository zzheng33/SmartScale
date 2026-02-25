import pandas as pd
import os

benchmarks = ['minisweep', 'lbm', 'cloverleaf', 'tealeaf']
base_dir = '/home/ac.zzheng/power/GPGPU/data/H100/spec_power_motif'

# Expected total power caps - apply these to every 13 rows
expected_caps = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

for benchmark in benchmarks:
    csv_file = os.path.join(base_dir, benchmark, 'runtime.csv')

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        continue

    # Read the CSV
    df = pd.read_csv(csv_file)

    print(f"\n{benchmark}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Before - unique power_caps: {sorted(df['power_cap'].unique())}")

    # Replace power_cap values: every 13 rows gets [800, 900, ..., 2000]
    new_power_caps = []
    for i in range(len(df)):
        # Determine which of the 13 values to use based on position
        cap_index = i % 13
        new_power_caps.append(expected_caps[cap_index])

    df['power_cap'] = new_power_caps

    # Save back to CSV
    df.to_csv(csv_file, index=False)

    print(f"  After - unique power_caps: {sorted(df['power_cap'].unique())}")
    print(f"  Sample rows:")
    print(f"    Row 1-3: {df.iloc[0:3]['power_cap'].tolist()}")
    print(f"    Row 13-15: {df.iloc[12:15]['power_cap'].tolist()}")

print("\nDone!")
