"""Unit test script for NPUDeviceTracker.

This script runs a YOLO11m model inference to stress the NPU while tracking
power, utilization, and memory usage. Results are saved to `npu_metrics.json`.
"""

import json
import pprint
import time

import mbltml
import numpy as np
import pytest

from mblt_tracker import NPUDeviceTracker

try:
    from qbruntime import Accelerator

    Accelerator()
except Exception as exc:
    pytest.skip(
        f"Skipping NPU integration test: NPU is not available: {exc}",
        allow_module_level=True,
    )

vision = pytest.importorskip(
    "mblt_model_zoo.vision", reason="NPU integration test requires mblt_model_zoo"
)
tqdm_module = pytest.importorskip("tqdm", reason="NPU integration test requires tqdm")

YOLO11m = vision.YOLO11m
tqdm = tqdm_module.tqdm


def _expected_available_all_mode_rails() -> list[str]:
    """Return all-mode rails expected to be available on the current device."""
    rails = ["npu", "ddr", "pmic"]
    device_count = mbltml.mbltmlGetDeviceCount()
    aries_type = getattr(mbltml, "MBLTML_DEVICE_ARIES", None)
    aries_devices = sum(
        1
        for dev_no in range(device_count)
        if mbltml.mbltmlGetDeviceType(dev_no) == aries_type
    )
    if aries_devices >= 4:
        rails.append("goldfinger")
    return rails


def test_npu_all_rail_tracking_records_npu_samples() -> None:
    tracker = NPUDeviceTracker(interval=1.0, rail_metrics="all")
    model = YOLO11m()
    x = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)
    expected_sampled_rails = _expected_available_all_mode_rails()

    for _ in range(3):
        model(x)

    tracker.start()
    start_time = time.time()
    try:
        while time.time() - start_time < 8.5:
            model(x)
    finally:
        tracker.stop()

    metrics = tracker.get_metric()
    assert metrics["samples"] > 0
    assert metrics["rail_metrics"]["selected"] == [
        "npu",
        "ddr",
        "pmic",
        "goldfinger",
    ]
    for rail in expected_sampled_rails:
        assert metrics[f"{rail}_rail_power_w_samples"] > 0


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
