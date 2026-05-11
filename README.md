# Mobilint Device Tracker

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-tracker/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
<p>
    <b>A lightweight Python library and CLI for tracking dynamic hardware metrics and collecting static system metadata across CPU, DRAM, GPU, and NPU.</b>
</p>
</div>
<!-- markdownlint-enable MD033 -->

## Overview

**mblt-tracker** is designed to help developers and researchers measure hardware performance with fair and consistent criteria. It provides a unified interface to poll metrics in the background while your code runs, producing both summarized statistics and detailed time-series traces.

### ✨ Key Features

- **Multi-Backend Support**: Unified interface for Intel CPU/DRAM, NVIDIA GPU, and Mobilint NPU.
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
from mblt_tracker import CPUDeviceTracker # or DRAMDeviceTracker, GPUDeviceTracker, NPUDeviceTracker

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

The following examples show representative `mblt-tracker collect` outputs across
Windows and Linux systems. Public static output intentionally omits
privacy-sensitive host and device instance identifiers: DRAM DIMM
serial/part/manufacturer details are not collected, and PCIe `bus_address` /
Windows `pnp_device_id` are not exposed, including when `--all-pcie-devices` is
used.

#### Windows host with Intel UHD Graphics, NVIDIA RTX 3090, and Mobilint NPU

```bash
$ mblt-tracker collect
WARNING:root:imports error
 You need to install pymongo>=3.9.0 in order to use MongoOutput
WARNING:root:imports error
  You need to install pandas>=0.25.1 in order to use DataFrameOutput
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
      "available_bytes": 15008358400,
      "configured_speed_mhz": 5600,
      "ram_type": "DDR5",
      "speed_mhz": 5600,
      "theoretical_bandwidth_gbps": 89.6,
      "total_bytes": 34113015808
    },
    "gpus": [
      {
        "class": "0x038000",
        "dev_no": 0,
        "device_id": "0x4680",
        "driver_date": "/Date(1764547200000)/",
        "driver_description": "Intel(R) UHD Graphics 770",
        "driver_provider": "Intel Corporation",
        "driver_version": "32.0.101.7082",
        "manufacturer": "Intel Corporation",
        "name": "Intel(R) UHD Graphics 770",
        "revision": "0x0c",
        "status": "OK",
        "subsystem_device_id": "0x7d96",
        "subsystem_vendor_id": "0x1462",
        "vendor_id": "0x8086"
      },
      {
        "architecture": "Ampere",
        "class": "0x030000",
        "dev_no": 0,
        "device_id": "0x2204",
        "driver_date": "/Date(1773705600000)/",
        "driver_description": "NVIDIA GeForce RTX 3090",
        "driver_provider": "NVIDIA",
        "driver_version": "595.97",
        "lane_width": "x4",
        "link_generation": "Gen1",
        "manufacturer": "NVIDIA",
        "memory_total_bytes": 25769803776,
        "name": "NVIDIA GeForce RTX 3090",
        "revision": "0xa1",
        "status": "OK",
        "subsystem_device_id": "0x1454",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      }
    ],
    "npus": [
      {
        "class": "0x078000",
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

#### Linux host with NVIDIA RTX PRO 6000 Blackwell GPUs and MLA100

```bash
$ mblt-tracker collect
WARNING:root:imports error
 You need to install pymongo>=3.9.0 in order to use MongoOutput
WARNING:root:imports error
  You need to install pandas>=0.25.1 in order to use DataFrameOutput
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
      "available_bytes": 321193263104,
      "total_bytes": 405389791232
    },
    "gpus": [
      {
        "class": "0x030000",
        "dev_no": 0,
        "device_id": "0x2000",
        "manufacturer": "ASPEED Technology, Inc.",
        "name": "ASPEED Graphics Family",
        "revision": "0x52",
        "subsystem_device_id": "0x1c6b",
        "subsystem_vendor_id": "0x15d9",
        "vendor_id": "0x1a03"
      },
      {
        "architecture": "Blackwell",
        "class": "0x030000",
        "dev_no": 0,
        "device_id": "0x2bb1",
        "driver_version": "580.95.05",
        "lane_width": "x16",
        "link_generation": "Gen1",
        "manufacturer": "NVIDIA Corporation",
        "memory_total_bytes": 102641958912,
        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "revision": "0xa1",
        "subsystem_device_id": "0x204b",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      },
      {
        "architecture": "Blackwell",
        "class": "0x030000",
        "dev_no": 1,
        "device_id": "0x2bb1",
        "driver_version": "580.95.05",
        "lane_width": "x16",
        "link_generation": "Gen5",
        "manufacturer": "NVIDIA Corporation",
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
        "card_id": 0,
        "card_model": "MLA100",
        "class": "0x7800002",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 0,
        "device_id": "0x0",
        "firmware": {
          "revision": "0",
          "version": "1.1"
        },
        "lane_width": "8",
        "link_generation": "4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x1093",
        "subsystem_vendor_id": "0x401",
        "vendor_id": "0x209F"
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
    "driver": {
      "aries_version": "1.12.0",
      "regulus_version": "N/A"
    },
    "gpu": {
      "cuda_driver": {
        "version": "13.0"
      },
      "driver": {
        "version": "580.95.05"
      }
    },
    "npu_driver_version": "1.12.0",
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

#### Linux host with MLA400 NPUs and NVML unavailable

```bash
$ mblt-tracker collect
WARNING:root:imports error
 You need to install pymongo>=3.9.0 in order to use MongoOutput
WARNING:root:imports error
  You need to install pandas>=0.25.1 in order to use DataFrameOutput
Warning: NVML not available. GPU information will not be collected.
{
  "hardware": {
    "cpu": {
      "architecture": "x86_64",
      "logical_cores": 16,
      "model_name": "11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz",
      "physical_cores": 8,
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 15901192192,
      "total_bytes": 67178881024
    },
    "npus": [
      {
        "board_name": "aries0",
        "card_id": 0,
        "card_model": "MLA400",
        "class": "0x7800002",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 0,
        "device_id": "0x0",
        "firmware": {
          "revision": "0",
          "version": "1.2.5"
        },
        "lane_width": "8",
        "link_generation": "4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
        "board_name": "aries1",
        "card_id": 0,
        "card_model": "MLA400",
        "class": "0x7800002",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 1,
        "device_id": "0x0",
        "firmware": {
          "revision": "0",
          "version": "1.2.5"
        },
        "lane_width": "8",
        "link_generation": "4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
        "board_name": "aries2",
        "card_id": 0,
        "card_model": "MLA400",
        "class": "0x7800002",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 2,
        "device_id": "0x0",
        "firmware": {
          "revision": "0",
          "version": "1.2.5"
        },
        "lane_width": "8",
        "link_generation": "4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
        "board_name": "aries3",
        "card_id": 0,
        "card_model": "MLA400",
        "class": "0x7800002",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "8",
        "dev_no": 3,
        "device_id": "0x0",
        "firmware": {
          "revision": "0",
          "version": "1.2.5"
        },
        "lane_width": "8",
        "link_generation": "4",
        "manufacturer": "Mobilint, Inc.",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      }
    ]
  },
  "inference": {
    "cpu": {
      "governor": "powersave",
      "max_processor_state_pct": null,
      "min_processor_state_pct": null,
      "power_plan": null
    },
    "cuda": {
      "version": "12.8"
    },
    "driver": {
      "aries_version": "1.12.0",
      "regulus_version": "N/A"
    },
    "npu_driver_version": "1.12.0",
    "os": {
      "kernel_version": "6.17.0-20-generic",
      "name": "Linux",
      "version": "Ubuntu 24.04.2 LTS"
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

| Metric | Intel CPU | Host DRAM | NVIDIA GPU | Mobilint NPU |
| :--- | :---: | :---: | :---: | :---: |
| **Power (W)** | ✅ (RAPL) | ✅ (RAPL DRAM) | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Utilization (%)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Memory (MB/%)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Temperature (C)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mobilint-cli`) |
| **Static Info** | ✅ Host/OS/DRAM | ✅ Host/OS/DRAM | ✅ NVML + PCIe | ✅ PCIe + `mobilint-cli` |
| **Per-Device Stats** | ✅ (Sockets) | ✅ (Sockets) | ✅ (GPU Indices) | ✅ (Logical NPU Cards) |

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

### Host DRAM

Uses the **Intel RAPL DRAM domain** through `pyRAPL` for host DRAM power measurements.

- **Platform/Hardware**: Requires a host that exposes RAPL DRAM energy counters.
- **Permission**: Requires read access to Intel RAPL sysfs, similar to CPU power tracking.
- **Features**: Tracks all detected CPU socket DRAM domains by default, or specific socket IDs with `DRAMDeviceTracker(socket_id=0)` / `DRAMDeviceTracker(socket_id=[0, 1])`.
- **Metrics**: Reports total host DRAM power through standard keys (`avg_power_w`, `p99_power_w`, `max_power_w`) and DRAM-specific aliases (`avg_dram_power_w`, `p99_dram_power_w`, `max_dram_power_w`). Per-socket statistics are returned under `metrics["dram"]`.
- **Trace**: `DRAMDeviceTracker.get_trace()` returns total host DRAM power as `list[(timestamp, power_w)]`.
- **Static Info**: `DRAMDeviceTracker.get_static_info()` returns the same privacy-first host CPU, aggregate DRAM capacity, and OS metadata as host static collection. Individual DIMM identifiers are not collected.

### Mobilint NPU

Polls the `mobilint-cli status -q` command, with a legacy JSON fallback for older environments.

- **Platform**: Currently supports **Linux only**.
- **Requirement**: Ensure [Mobilint Utility Tool](https://docs.mobilint.com/v1.0/en/installing_utility.html) is installed and `mobilint-cli` is in your PATH.
- **Device Selection**: Tracks all logical NPU cards by default, or selected logical card IDs with `NPUDeviceTracker(npu_id=0)` / `NPUDeviceTracker(npu_id=[0, 1])`.
- **MLA100 vs MLA400**: `status -q` output is classified best-effort. Devices with a `Power.GOLDFINGER` rail are treated as MLA400 and grouped as one logical card. MLA100 devices remain one logical card per PCIe card. PCIe subsystem IDs are also used as a fallback (`0x401/0x1093` for MLA100, `0x402/0x108B` for MLA400 observed outputs).
- **NPU Power**: Distinguishes between NPU core power and total card/system power. For MLA400, `Power.Total` is reported by the first chip, while NPU core, memory, and utilization are aggregated across the grouped Aries chips.
- **DDR/PMIC/GOLDFINGER Power**: Parses optional NPU board DDR, PMIC, and MLA400 GOLDFINGER power rails when present.
- **Temperature**: Parses NPU temperature from `mobilint-cli status` output when available.
- **Static Info**: Reports Mobilint PCIe device information and parses driver, firmware, product, board, `card_model`, and `card_id` metadata from `mobilint-cli status` when available.

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
- **DRAM**: DRAM-specific power keys include `avg_dram_power_w`, `p99_dram_power_w`, and `max_dram_power_w`. `dram` contains per-socket statistics keyed by socket ID.
- **NPU**: NPU-specific power keys include `avg_npu_power_w`, `p99_npu_power_w`, `max_npu_power_w`, `avg_ddr_power_w`, `p99_ddr_power_w`, `max_ddr_power_w`, `avg_pmic_power_w`, `p99_pmic_power_w`, `max_pmic_power_w`, `avg_goldfinger_power_w`, `p99_goldfinger_power_w`, `max_goldfinger_power_w`, `avg_total_power_w`, `p99_total_power_w`, and `max_total_power_w`. `avg_power_w` is mapped to total power for cross-device consistency. `npu` contains per logical card statistics keyed by card ID.

### Time-Series Trace APIs

All trackers expose trace APIs for post-processing and plotting:

```python
tracker.get_trace()       # Power trace: list[(timestamp, power_w)]
tracker.get_util_trace()  # Utilization trace: list[(timestamp, utilization_pct)]
tracker.get_temp_trace()  # Temperature trace: list[(timestamp, temperature_c)]
```

NPU trackers additionally expose rail-specific power traces:

```python
tracker.get_npu_power_trace()         # NPU core power
tracker.get_ddr_power_trace()         # On-board NPU DDR power, when available
tracker.get_pmic_power_trace()        # NPU PMIC power, when available
tracker.get_goldfinger_power_trace()  # MLA400 GOLDFINGER input power, when available
```

---

## 🔍 Static Information

For benchmark reproducibility, each tracker exposes `get_static_info()`:

```python
from mblt_tracker import CPUDeviceTracker, GPUDeviceTracker, NPUDeviceTracker

tracker = CPUDeviceTracker()
info = tracker.get_static_info()
```

Static information is collected on a best-effort, privacy-first basis and may vary by platform and permissions.

Typical fields include:

- `hardware.cpu`: CPU architecture, model name, vendor, physical cores, logical cores
- `hardware.dram`: total and available memory in bytes, plus optional privacy-safe aggregate fields such as `ram_type`, `speed_mhz`, `configured_speed_mhz`, and `theoretical_bandwidth_gbps` when available. Individual DIMM serial numbers, part numbers, manufacturers, and `hardware.dram.dimms` are not collected or exposed.
- `inference.os`: OS name, version, and kernel version
- `inference.cpu`: OS-independent CPU power policy object. Linux fills `governor`; Windows fills `power_plan`, `min_processor_state_pct`, and `max_processor_state_pct`. Unavailable OS-specific attributes are kept as `null`.
- `hardware.gpu`: `GPUDeviceTracker.get_static_info()` output with `device_count` and a `devices` list containing tracked GPU indices and names
- `hardware.gpus`: `mblt-tracker collect` output containing NVML-discovered NVIDIA GPU devices enriched with PCIe vendor/device IDs and link information where available. Private PCIe instance identifiers such as `bus_address` and `pnp_device_id` are omitted from public output.
- `hardware.npus`: Mobilint PCIe devices, including vendor/device IDs, link information, and firmware metadata where available. Private PCIe instance identifiers such as `bus_address` and `pnp_device_id` are omitted from public output.
- `hardware.npus[].card_model`: best-effort Mobilint card model classification such as `MLA100` or `MLA400` when `mobilint-cli status -q` exposes enough information
- `hardware.npus[].card_id`: logical NPU card ID used by `NPUDeviceTracker(npu_id=...)`; MLA400 Aries chips share the same card ID
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
