"""Unit test script for CPUDeviceTracker.

This script performs matrix multiplications to stress the CPU while tracking
power, utilization, and memory usage. Results are saved to `cpu_metrics.json`.
"""

import json
import pprint
import time

import numpy as np
from tqdm import tqdm

from mblt_tracker import CPUDeviceTracker

if __name__ == "__main__":
    tracker = CPUDeviceTracker(interval=0.1)

    x = np.random.rand(1000, 1000)
    y = np.random.rand(1000, 1000)

    # warm up
    for _ in range(10):
        z = np.matmul(x, y)

    tracker.start()
    start_time = time.time()
    duration: int = 60
    current_sec: float = 0
    with tqdm(
        total=duration,
        desc="Running CPU Test",
        bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
    ) as pbar:
        while current_sec < duration:
            z = np.matmul(x, y)
            current_sec = time.time() - start_time
            # Calculate update delta and cast to satisfy static analysis inference
            update_delta = float(current_sec) - float(pbar.n)
            if update_delta > 0:
                pbar.update(update_delta)
    tracker.stop()

    pprint.pprint(tracker.get_metric())
    with open("cpu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tracker.get_metric(), f, indent=4)
