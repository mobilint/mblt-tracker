"""Unit test script for NPUDeviceTracker.

This script runs a YOLO11m model inference to stress the NPU while tracking
power, utilization, and memory usage. Results are saved to `npu_metrics.json`.
"""

import json
import pprint
import time

import numpy as np
from mblt_model_zoo.vision import YOLO11m
from tqdm import tqdm

from mblt_tracker import NPUDeviceTracker

if __name__ == "__main__":
    tracker = NPUDeviceTracker(interval=0.1)

    model = YOLO11m()
    x = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)

    # warm up
    for _ in range(10):
        model(x)

    tracker.start()
    duration: int = 60
    current_sec: float = 0
    start_time = time.time()
    with tqdm(
        total=duration,
        desc="Running NPU Test",
        bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]",
    ) as pbar:
        while current_sec < duration:
            model(x)
            current_sec = time.time() - start_time
            # Calculate update delta and cast to satisfy static analysis inference
            update_delta = float(current_sec) - float(pbar.n)
            if update_delta > 0:
                pbar.update(update_delta)
    tracker.stop()

    pprint.pprint(tracker.get_metric())
    with open("npu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tracker.get_metric(), f, indent=4)
