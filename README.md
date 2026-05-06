# Mobilint Device Tracker

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-tracker/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
<p>
    <b>A lightweight Python library and CLI for tracking dynamic hardware metrics and collecting static system metadata across CPU, GPU, and NPU.</b>
</p>
</div>
<!-- markdownlint-enable MD033 -->

## Overview

**mblt-tracker** is designed to help developers and researchers measure hardware performance with fair and consistent criteria. It provides a unified interface to poll metrics in the background while your code runs, producing both summarized statistics and detailed time-series traces.

### ✨ Key Features

- **Multi-Backend Support**: Unified interface for Intel CPU, NVIDIA GPU, and Mobilint NPU.
- **Background Tracking**: Uses a background scheduler to poll metrics without blocking your main execution.
- **Comprehensive Metrics**: Capture Power (Watts), Utilization (%), Memory Usage (MB/%), and Temperature (C).
- **Statistical Summaries**: Automatically calculates averages, peaks (max), and p99 values.
- **Time-Series Traces**: Export raw data for custom plotting and analysis.
- **Static Metadata**: Collect best-effort host, OS, PCIe, driver, firmware, and device information for reproducible benchmarks.
- **CLI Collection Tool**: Use `mblt-tracker collect` to export static host and PCIe information as JSON.
- **PCIe Discovery**: Detect GPU/NPU-related PCIe devices on Linux and Windows, including link speed/width where available.
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

def format_metric(value, unit):
    return f"{value:.2f} {unit}" if value is not None else f"N/A {unit}"

print(f"Average Power: {format_metric(metrics['avg_power_w'], 'W')}")
print(f"Max Utilization: {format_metric(metrics['max_utilization_pct'], '%')}")
print(f"Max Temperature: {format_metric(metrics['max_temperature_c'], 'C')}")

# 5. Export time-series traces
power_trace = tracker.get_trace()      # list of (timestamp, power_w)
util_trace = tracker.get_util_trace()  # list of (timestamp, utilization_pct)
temp_trace = tracker.get_temp_trace()  # list of (timestamp, temperature_c)

# 6. Collect static metadata for reproducibility
static_info = tracker.get_static_info()
```

---

## Command Line Interface

`mblt-tracker` provides a CLI for collecting static host and PCIe metadata without writing Python code.

```bash
# Print collected information to stdout
mblt-tracker collect

# Save collected information as JSON
mblt-tracker collect -o static-info.json

# Include all PCIe devices instead of only GPU/NPU-related devices
mblt-tracker collect --all-pcie-devices

# Filter NPU PCIe discovery by vendor/device/class
mblt-tracker collect --pcie-vendor-id 0x1ed5
mblt-tracker collect --pcie-vendor-id 1ed5 --pcie-device-id 0100
mblt-tracker collect --pcie-class-filter 0x12
```

The CLI output is a JSON document containing best-effort host CPU, DRAM, OS, GPU, NPU, driver, and PCIe information. NVIDIA GPU entries are sourced from NVML and enriched with PCIe metadata where available. On Linux, PCIe information is read from sysfs. On Windows, PCI devices are collected through PowerShell/CIM/PnP queries.

### Example Output

#### Windows

```powershell
> mblt-tracker collect
{
  "hardware": {
    "cpu": {
      "architecture": "AMD64",
      "logical_cores": 20,
      "model_name": "13th Gen Intel(R) Core(TM) i5-13500",
      "physical_cores": 14,
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 14174498816,
      "dimms": [
        {
          "capacity_bytes": 17179869184,
          "configured_speed_mhz": 5600,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M323R2GA3PB0-CWMOL",
          "serial_number": "48A201A4",
          "speed_mhz": 5600,
          "total_width_bits": 64,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 17179869184,
          "configured_speed_mhz": 5600,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M323R2GA3PB0-CWMOL",
          "serial_number": "48A201E5",
          "speed_mhz": 5600,
          "total_width_bits": 64,
          "type": "DDR5"
        }
      ],
      "theoretical_bandwidth_gbps": 89.6,
      "total_bytes": 34113015808
    },
    "gpus": [
      {
        "architecture": "Ampere",
        "bus_address": "0000:03:00.0",
        "current_link_speed": "8.0 GT/s PCIe",
        "current_link_width": "4",
        "dev_no": 0,
        "device_id": "0x2204",
        "driver_date": "/Date(1773705600000)/",
        "driver_description": "NVIDIA GeForce RTX 3090",
        "driver_provider": "NVIDIA",
        "driver_version": "595.97",
        "lane_width": "x4",
        "link_generation": "Gen2",
        "manufacturer": "NVIDIA",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "16",
        "memory_total_bytes": 25769803776,
        "name": "NVIDIA GeForce RTX 3090",
        "pnp_device_id": "PCI\\VEN_10DE&DEV_2204&SUBSYS_145410DE&REV_A1\\4&126C804A&0&00E0",
        "revision": "0xa1",
        "status": "OK",
        "subsystem_device_id": "0x1454",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      }
    ],
    "npus": [
      {
        "bus_address": "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&3691B449&0&0008",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 0,
        "device_id": "0x0000",
        "driver_date": "/Date(1774828800000)/",
        "driver_description": "MOBILINT NPU Accelerator",
        "driver_provider": "MOBILINT, Inc.",
        "lane_width": "x8",
        "link_generation": "Gen4",
        "manufacturer": "MOBILINT, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "name": "MOBILINT NPU Accelerator",
        "pnp_device_id": "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&3691B449&0&0008",
        "revision": "0x02",
        "status": "OK",
        "subsystem_device_id": "0x1093",
        "subsystem_vendor_id": "0x0402",
        "vendor_id": "0x209f"
      }
    ]
  },
  "inference": {
    "cpu": {
      "governor": null,
      "max_processor_state_pct": 100,
      "min_processor_state_pct": 100,
      "power_plan": "High performance"
    },
    "cuda": {
      "version": "not_found"
    },
    "gpu": {
      "cuda_driver": {
        "version": "13.2"
      },
      "driver": {
        "version": "595.97"
      }
    },
    "npu_driver_version": "1.8.1.1348",
    "os": {
      "kernel_version": "11",
      "name": "Windows",
      "version": "10.0.26200"
    },
    "qbcompiler": {
      "version": "not_installed"
    },
    "qbruntime": {
      "version": "v1.2.0"
    }
  }
}
```

#### Linux

```bash
$ mblt-tracker collect
[sudo] password for dmidecode:
{
  "hardware": {
    "cpu": {
      "architecture": "x86_64",
      "logical_cores": 96,
      "model_name": "INTEL(R) XEON(R) GOLD 6542Y",
      "physical_cores": 48,
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 326259752960,
      "dimms": [
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01233104F96929",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01232804E78C2F",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01232804E78A42",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01232804E78C30",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01232804E65B3B",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        },
        {
          "capacity_bytes": 68719476736,
          "configured_speed_mhz": 4800,
          "data_width_bits": 64,
          "manufacturer": "Samsung",
          "part_number": "M321R8GA0BB0-CQKZJ",
          "serial_number": "80CE01232804E78A46",
          "speed_mhz": 4800,
          "total_width_bits": 80,
          "type": "DDR5"
        }
      ],
      "theoretical_bandwidth_gbps": 230.4,
      "total_bytes": 405389791232
    },
    "gpus": [
      {
        "architecture": "Blackwell",
        "bus_address": "0000:17:00.0",
        "class": "0x030000",
        "current_link_speed": "2.5 GT/s PCIe",
        "current_link_width": "16",
        "dev_no": 0,
        "device_id": "0x2bb1",
        "driver_version": "580.95.05",
        "lane_width": "x16",
        "link_generation": "Gen1",
        "manufacturer": "NVIDIA Corporation",
        "max_link_speed": "32.0 GT/s PCIe",
        "max_link_width": "16",
        "memory_total_bytes": 102641958912,
        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "revision": "0xa1",
        "subsystem_device_id": "0x204b",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      },
      {
        "architecture": "Blackwell",
        "bus_address": "0000:e1:00.0",
        "class": "0x030000",
        "current_link_speed": "2.5 GT/s PCIe",
        "current_link_width": "16",
        "dev_no": 1,
        "device_id": "0x2bb1",
        "driver_version": "580.95.05",
        "lane_width": "x16",
        "link_generation": "Gen1",
        "manufacturer": "NVIDIA Corporation",
        "max_link_speed": "32.0 GT/s PCIe",
        "max_link_width": "16",
        "memory_total_bytes": 102641958912,
        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "revision": "0xa1",
        "subsystem_device_id": "0x204b",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      }
    ],
    "npus": [
      {
        "board_name": "aries0",
        "bus_address": "0000:bd:00.0",
        "class": "0x078000",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 0,
        "device_id": "0x0000",
        "firmware": {
          "version": "fb9a5980"
        },
        "lane_width": "x8",
        "link_generation": "Gen4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "name": "Aries",
        "revision": "0x02",
        "subsystem_device_id": "0x1093",
        "subsystem_vendor_id": "0x0401",
        "vendor_id": "0x209f"
      }
    ]
  },
  "inference": {
    "cpu": {
      "governor": "schedutil",
      "max_processor_state_pct": null,
      "min_processor_state_pct": null,
      "power_plan": null
    },
    "cuda": {
      "version": "12.8"
    },
    "gpu": {
      "cuda_driver": {
        "version": "13.0"
      },
      "driver": {
        "version": "580.95.05"
      }
    },
    "os": {
      "kernel_version": "6.8.0-110-generic",
      "name": "Linux",
      "version": "Ubuntu 24.04 LTS"
    },
    "qbcompiler": {
      "version": "not_installed"
    },
    "qbruntime": {
      "version": "v1.2.0"
    }
  }
}
```

---

## Metrics Coverage

| Metric | Intel CPU | NVIDIA GPU | Mobilint NPU |
| :--- | :---: | :---: | :---: |
| **Power (W)** | ✅ (RAPL) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Utilization (%)** | ✅ (`psutil`) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Memory (MB/%)** | ✅ (`psutil`) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Temperature (C)** | ✅ (`psutil`) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Static Info** | ✅ Host/OS/DRAM | ✅ NVML + PCIe | ✅ PCIe + `mobilint-cli` |
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
- **Temperature**: Uses `psutil.sensors_temperatures()` when the platform exposes CPU thermal sensors.
- **Static Info**: Reports CPU architecture, model, vendor, physical/logical cores, DRAM capacity, OS details, and Linux CPU governor when available.

### NVIDIA GPU

Uses **NVML** (via `nvidia-ml-py`) for high-fidelity hardware monitoring.

- **Features**: Tracks total system GPU usage or specific indices (e.g., `GPUDeviceTracker(gpu_id=[0, 1])`).
- **Dependencies**: Requires NVIDIA Drivers and NVML library installed.
- **Temperature**: Reads on-die GPU temperature through NVML.
- **Static Info**: `GPUDeviceTracker.get_static_info()` reports the detected GPU count, tracked device names, NVIDIA driver version, and the raw NVML CUDA driver version. The `mblt-tracker collect` CLI provides richer NVML-discovered GPU metadata enriched with PCIe device/link information when available.

### Mobilint NPU

Polls the `mobilint-cli status` command.

- **Platform**: Currently supports **Linux only**.
- **Requirement**: Ensure [Mobilint Utility Tool](https://docs.mobilint.com/v1.0/en/installing_utility.html) is installed and `mobilint-cli` is in your PATH.
- **NPU Power**: Distinguishes between NPU-specific power and total system power.
- **Temperature**: Parses NPU temperature from `mobilint-cli status` output when available.
- **Static Info**: Reports Mobilint PCIe device information and parses driver, firmware, product, and board metadata from `mobilint-cli status` when available.

---

## 📝 Metric Output Format

Calling `get_metric()` returns a dictionary with standardized cross-device keys where applicable. Missing or unavailable measurements are returned as `None`.

```json
{
  "avg_power_w": 25.4,
  "p99_power_w": 40.1,
  "max_power_w": 45.2,
  "avg_utilization_pct": 78.5,
  "p99_utilization_pct": 90.0,
  "max_utilization_pct": 95.0,
  "avg_memory_used_mb": 2048.0,
  "p99_memory_used_mb": 3072.0,
  "max_memory_used_mb": 4096.0,
  "total_memory_mb": 8192.0,
  "avg_memory_used_pct": 25.0,
  "p99_memory_used_pct": 37.5,
  "max_memory_used_pct": 50.0,
  "avg_temperature_c": 72.3,
  "p99_temperature_c": 79.0,
  "max_temperature_c": 80.0,
  "samples": 100,
  "util_samples": 101
}
```

Tracker-specific fields may also be present:

- **CPU**: `cpu` contains per-socket statistics keyed by socket ID.
- **GPU**: `gpu` contains per-GPU statistics keyed by GPU index. GPU-specific summary keys include `avg_gpu_util_pct`, `p99_gpu_util_pct`, `max_gpu_util_pct`, `avg_mem_util_pct`, and `p99_mem_util_pct`.
- **NPU**: NPU-specific power keys include `avg_npu_power_w`, `p99_npu_power_w`, `max_npu_power_w`, `avg_total_power_w`, `p99_total_power_w`, and `max_total_power_w`. `avg_power_w` is mapped to total power for cross-device consistency.

### Time-Series Trace APIs

All trackers expose trace APIs for post-processing and plotting:

```python
tracker.get_trace()       # Power trace: list[(timestamp, power_w)]
tracker.get_util_trace()  # Utilization trace: list[(timestamp, utilization_pct)]
tracker.get_temp_trace()  # Temperature trace: list[(timestamp, temperature_c)]
```

---

## 🔍 Static Information

For benchmark reproducibility, each tracker exposes `get_static_info()`:

```python
from mblt_tracker import CPUDeviceTracker, GPUDeviceTracker, NPUDeviceTracker

tracker = CPUDeviceTracker()
info = tracker.get_static_info()
```

Static information is collected on a best-effort basis and may vary by platform and permissions.

Typical fields include:

- `hardware.cpu`: CPU architecture, model name, vendor, physical cores, logical cores
- `hardware.dram`: total and available memory in bytes
- `hardware.dram.dimms`: physical DIMM metadata from `dmidecode` when available. On Linux, sudo password is required for the CLI to collect this interactively.
- `inference.os`: OS name, version, and kernel version
- `inference.cpu`: OS-independent CPU power policy object. Linux fills `governor`; Windows fills `power_plan`, `min_processor_state_pct`, and `max_processor_state_pct`. Unavailable OS-specific attributes are kept as `null`.
- `hardware.gpu`: `GPUDeviceTracker.get_static_info()` output with `device_count` and a `devices` list containing tracked GPU indices and names
- `hardware.gpus`: `mblt-tracker collect` output containing NVML-discovered NVIDIA GPU devices enriched with PCIe vendor/device IDs and link information where available
- `hardware.npus`: Mobilint PCIe devices, including vendor/device IDs, link information, and firmware metadata where available
- `inference.gpu`: NVIDIA driver and CUDA driver versions. The CLI normalizes the CUDA driver version as a string such as `"13.0"`; `GPUDeviceTracker.get_static_info()` returns the raw NVML CUDA driver integer.
- `hardware.npus[].firmware`: per-NPU firmware metadata where available. Linux currently maps `mobilint-cli status` firmware rows by device order.
- `inference.npu_driver_version`: host Mobilint NPU driver version when available. Linux may also include `inference.driver` with Aries/Regulus driver metadata parsed from `mobilint-cli status`.

PCIe discovery supports:

- **Linux**: `/sys/bus/pci/devices`
- **Windows**: PowerShell/CIM/PnP PCI device queries

For tests or custom environments, `MBLT_TRACKER_PCI_SYSFS` can override the Linux PCI sysfs root. `NPUDeviceTracker.get_static_info()` can customize NPU PCIe matching with `MBLT_TRACKER_NPU_PCI_VENDOR_ID`, `MBLT_TRACKER_NPU_PCI_DEVICE_ID`, and `MBLT_TRACKER_NPU_PCI_CLASS_FILTER`. The `mblt-tracker collect` CLI uses the corresponding `--pcie-vendor-id`, `--pcie-device-id`, and `--pcie-class-filter` flags.

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
