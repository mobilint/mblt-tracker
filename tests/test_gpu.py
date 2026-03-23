"""Unit test script for GPUDeviceTracker.

This script performs intensive matrix multiplications using PyTorch to stress
the GPU while tracking power, utilization, and memory usage. Results are
saved to `gpu_metrics.json`.
"""

import json
import pprint
import time

import torch
from torch import cuda, matmul
from tqdm import tqdm

from mblt_tracker import GPUDeviceTracker

if __name__ == "__main__":
    tracker = GPUDeviceTracker(interval=0.1)

    x: torch.Tensor = torch.rand(10000, 10000).to("cuda")
    y: torch.Tensor = torch.rand(10000, 10000).to("cuda")

    # warm up
    for _ in range(10):
        z = matmul(x, y)

    cuda.synchronize()
    tracker.start()
    start_time = time.time()
    duration: int = 60
    current_sec: int = 0
    with torch.no_grad():
        with tqdm(
            total=duration,
            desc="Running GPU Test",
            bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
        ) as pbar:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    # Final update to reach the end, with explicit casts for static analysis
                    remaining = int(duration) - int(current_sec)
                    if remaining > 0:
                        pbar.update(remaining)
                    break

                new_sec = int(elapsed)
                if new_sec > current_sec:
                    # Use explicit int() casts to satisfy Pyre2 inference
                    pbar.update(int(new_sec) - int(current_sec))
                    current_sec = new_sec

                z = matmul(x, y)
                cuda.synchronize()
    tracker.stop()

    pprint.pprint(tracker.get_metric())
    with open("gpu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tracker.get_metric(), f, indent=4)
