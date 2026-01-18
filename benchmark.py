import json
import os
import numpy as np
import scipy
import shutil
import subprocess


def run_dgemm(kernel, A, B, n_trials=10):
    # Clean up and create directories
    if os.path.exists("input"):
        shutil.rmtree("input")
    if os.path.exists("output"):
        shutil.rmtree("output")

    os.makedirs("input")
    os.makedirs("output")

    # Save input data
    np.save("input/A.npy", A)
    np.save("input/B.npy", B)

    # Run the kernel with correct arguments
    cmd = [
        f"./{kernel}",
        "--input",
        "input",
        "--output",
        "output",
        "--trial-max",
        str(n_trials),
        "--time-max",
        "inf",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {kernel}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Kernel {kernel} execution failed")

    C = np.load("output/C.npy")
    with open("output/measurements.json", "r") as f:
        measurements = dict(json.load(f))
    measurements["C"] = C
    return measurements


if __name__ == "__main__":
    # The reference machine is Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz (turbo boost is disabled)
    # Manual: https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html
    # 8 wide register x 2 FMA units
    max_speed_gflops = 2.7 * 8 * 2
    test_sizes = [
        31,
        32,
        96,
        97,
        127,
        128,
        129,
        191,
        192,
        229,
        255,
        256,
        257,
        319,
        320,
        321,
        417,
        479,
        480,
        511,
        512,
        639,
        640,
        767,
        768,
        769,
    ]

    naive_kernel_name = "dgemm-naive"
    optimized_kernel_name = "dgemm-optimized"

    for n in test_sizes:
        # Double precision float
        # Ref: https://numpy.org/doc/stable/user/basics.types.html#relationship-between-numpy-data-types-and-c-data-types
        input_data_A = np.random.rand(n, n).astype(np.float64)
        input_data_B = np.random.rand(n, n).astype(np.float64)

        baseline_times = []
        optimized_times = []

        n_trials = 100
        for trial in range(n_trials):
            # if trial % 100 == 0:
            #     print(f"Progress: {trial}/{n_trials}")

            # Randomly decide order for this trial to avoid systemic bias
            order = np.random.choice(["baseline_first", "optimized_first"])

            if order == "baseline_first":
                baseline_result = run_dgemm(
                    naive_kernel_name, input_data_A, input_data_B, n_trials=1
                )
                baseline_times.append(baseline_result["time"])
                optimized_result = run_dgemm(
                    optimized_kernel_name, input_data_A, input_data_B, n_trials=1
                )
                optimized_times.append(optimized_result["time"])
                assert np.allclose(
                    baseline_result["C"], optimized_result["C"]
                ), "Results do not match!"
            else:
                optimized_result = run_dgemm(
                    optimized_kernel_name, input_data_A, input_data_B, n_trials=1
                )
                optimized_times.append(optimized_result["time"])
                baseline_result = run_dgemm(
                    naive_kernel_name, input_data_A, input_data_B, n_trials=1
                )
                baseline_times.append(baseline_result["time"])
                assert np.allclose(
                    baseline_result["C"], optimized_result["C"]
                ), "Results do not match!"

        baseline_times = np.array(baseline_times)
        optimized_times = np.array(optimized_times)

        wins = np.sum(optimized_times < baseline_times)
        win_rate = wins / n_trials

        # Use binomial test (null hypothesis: win rate = 0.5)
        p_value = scipy.stats.binomtest(wins, n_trials, 0.5).pvalue

        baseline_times_min = np.min(baseline_times)
        baseline_mflops = 2.0e-6 * n * n * n / (baseline_times_min * 1.0e-9)
        baseline_peak_perc = ((baseline_mflops / 1000) / max_speed_gflops) * 100

        optimized_times_min = np.min(optimized_times)
        optimized_mflops = 2.0e-6 * n * n * n / (optimized_times_min * 1.0e-9)
        optimized_peak_perc = ((optimized_mflops / 1000) / max_speed_gflops) * 100

        print(
            f"Size: {n}       Mflops: {optimized_mflops:.2f} (peak perc: {optimized_peak_perc:.6f}%)      speedup: {baseline_times_min / optimized_times_min:.2f}x        binomial test p-value: {p_value:.6f} (win rate: {win_rate:.1%})"
        )