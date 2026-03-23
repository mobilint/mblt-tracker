# Mobilint Device Tracker

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-tracker/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
<p>
    <b>A lightweight Python library for tracking hardware metrics (Power, Utilization, Memory) across CPU, GPU, and NPU.</b>
</p>
</div>
<!-- markdownlint-enable MD033 -->

## Overview

**mblt-tracker** is designed to help developers and researchers measure hardware performance with fair and consistent criteria. It provides a unified interface to poll metrics in the background while your code runs, producing both summarized statistics and detailed time-series traces.

### ✨ Key Features

- **Multi-Backend Support**: Unified interface for Intel CPU, NVIDIA GPU, and Mobilint NPU.
- **Background Tracking**: Uses a background scheduler to poll metrics without blocking your main execution.
- **Comprehensive Metrics**: Capture Power (Watts), Utilization (%), and Memory Usage (MB/%).
- **Statistical Summaries**: Automatically calculates averages, peaks (max), and p99 values.
- **Time-Series Traces**: Export raw data for custom plotting and analysis.
- **Lightweight**: Minimal overhead, designed for production and research environments.

---

## 🚀 Installation

[![PyPI - Version](https://img.shields.io/pypi/v/mblt-tracker?logo=pypi&logoColor=white)](https://pypi.org/project/mblt-tracker/)
[![PyPI Downloads](https://static.pepy.tech/badge/mblt-tracker?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/mblt-tracker)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mblt-tracker?logo=python&logoColor=gold)](https://pypi.org/project/mblt-tracker/)

```bash
pip install mblt-tracker
```

For the latest features, install directly from source:

```bash
git clone https://github.com/mobilint/mblt-tracker.git
cd mblt-tracker
pip install -e .
```

---

## 📖 Quick Start Guide

The typical workflow involves initializing a tracker, starting it before your target workload, and stopping it after.

```python
from mblt_tracker import CPUDeviceTracker # or GPUDeviceTracker, NPUDeviceTracker

# 1. Initialize with a polling interval (seconds)
tracker = CPUDeviceTracker(interval=0.1)

# 2. Start tracking (best to run after warm-up)
tracker.start()

# --- Your workload starts here ---
# e.g., model.inference(data)
# --- Your workload ends here ---

# 3. Stop tracking
tracker.stop()

# 4. Access results
metrics = tracker.get_metric()
print(f"Average Power: {metrics['avg_power_w']:.2f} W")
print(f"Max Utilization: {metrics['max_utilization_pct']:.2f} %")

# 5. Export time-series trace (list of (timestamp, power_w))
trace = tracker.get_trace()
```

---

## 📊 Metrics Coverage

| Metric | Intel CPU | NVIDIA GPU | Mobilint NPU |
| :--- | :---: | :---: | :---: |
| **Power (W)** | ✅ (RAPL) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Utilization (%)** | ✅ (`psutil`) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Memory (MB/%)** | ✅ (`psutil`) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Per-Device Stats** | ✅ (Sockets) | ✅ (GPU Indices) | ❌ (Global/Total) |

---

## 🛠️ Hardware Specifics

### Intel CPU

Uses **pyRAPL** for power measurements and **psutil** for utilization/memory.

- **Permission**: Requires read access to Intel RAPL sysfs.

  ```bash
  sudo chmod -R a+r /sys/class/powercap/intel-rapl/
  ```

- **Docker**: Run containers with `--privileged` or mount the powercap directory.

- **Features**: Tracks total system CPU usage or specific indices (e.g., `CPUDeviceTracker(cpu_id=[0, 1])`).

### NVIDIA GPU

Uses **NVML** (via `nvidia-ml-py`) for high-fidelity hardware monitoring.

- **Features**: Tracks total system GPU usage or specific indices (e.g., `GPUDeviceTracker(gpu_id=[0, 1])`).
- **Dependencies**: Requires NVIDIA Drivers and NVML library installed.

### Mobilint NPU

Polls the `mobilint-cli status` command.

- **Platform**: Currently supports **Linux only**.
- **Requirement**: Ensure [Mobilint Utility Tool](https://docs.mobilint.com/v1.0/en/installing_utility.html) is installed and `mobilint-cli` is in your PATH.
- **NPU Power**: Distinguishes between NPU-specific power and total system power.

---

## 📝 Metric Output Format

Calling `get_metric()` returns a dictionary with the following standard keys (where applicable):

```json
{
  "avg_power_w": 25.4,          // Average total power in Watts
  "max_power_w": 45.2,          // Peak power observed
  "p99_power_w": 40.1,          // 99th percentile power
  "avg_utilization_pct": 78.5,  // Average device utilization
  "max_utilization_pct": 95.0,  // Peak device utilization
  "avg_memory_used_mb": 2048.0, // Average memory usage
  "total_memory_mb": 8192.0,    // Total available memory
  "samples": 100                // Number of data points collected
}
```

*Note: Some trackers provide additional keys like `cpu` or `gpu` for per-socket/per-device breakdown.*

---

## 🤝 Contributing

We welcome contributions! To set up for development:

1. Install dev dependencies: `pip install -e ".[dev]"`
2. Run tests: `pytest tests/`

## 📄 License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE](https://github.com/mobilint/mblt-tracker/blob/master/LICENSE) file for details.

---

<!-- markdownlint-disable MD033 -->
<div align="center">
    <small>Developed with ❤️ by <a href="https://www.mobilint.com/">Mobilint Inc.</a></small>
</div>
<!-- markdownlint-enable MD033 -->