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
- **Static Metadata**: Collect best-effort host, OS, CPU, DRAM capacity/type/speed, motherboard, PCIe, driver, firmware, and device information for reproducible benchmarks.
- **CLI Collection Tool**: Use `mblt-tracker collect` to export static host and PCIe information as JSON.
- **PCIe Discovery**: Detect GPU/NPU-related PCIe devices on Linux and Windows, including current and maximum link speed/width where available.
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

The CLI output is a JSON document containing best-effort host CPU, DRAM, motherboard, OS, GPU, NPU, driver, and PCIe information. For NVIDIA GPU entries, NVML is the source of truth for GPU identity and PCIe link metadata; OS PCIe discovery is used only to attach non-link PCIe identifiers and descriptive fields where available. On Linux, non-NVIDIA PCIe information is read from sysfs. On Windows, non-NVIDIA PCI devices are collected through PowerShell/CIM/PnP queries.

### Example Output

The following examples show representative `mblt-tracker collect` outputs across
Windows and Linux systems. Public static output intentionally omits
privacy-sensitive host and device instance identifiers: DRAM DIMM part/serial
numbers, motherboard serial/asset tags, and PCIe `bus_address` / Windows
`pnp_device_id` are not exposed, including when `--all-pcie-devices` is used.
DRAM byte counts are accompanied by MB/GB display units; DRAM speed, type,
module capacity, and estimated theoretical bandwidth are kept to make benchmark results easier to
interpret.

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
      "model_name": "13th Gen Intel(R) Core(TM) i5-13500",
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 15580520448,
      "available_gb": 14.51,
      "available_mb": 14858.74,
      "configured_speed_mhz": 5600,
      "module_count": 2,
      "modules": [
        {
          "capacity_bytes": 17179869184,
          "capacity_gb": 16.0,
          "capacity_mb": 16384.0,
          "configured_speed_mhz": 5600,
          "data_width_bits": 64,
          "ram_type": "DDR5",
          "speed_mhz": 5600,
          "theoretical_bandwidth_gbps": 44.8,
          "total_width_bits": 64
        },
        {
          "capacity_bytes": 17179869184,
          "capacity_gb": 16.0,
          "capacity_mb": 16384.0,
          "configured_speed_mhz": 5600,
          "data_width_bits": 64,
          "ram_type": "DDR5",
          "speed_mhz": 5600,
          "theoretical_bandwidth_gbps": 44.8,
          "total_width_bits": 64
        }
      ],
      "ram_type": "DDR5",
      "speed_mhz": 5600,
      "theoretical_bandwidth_gbps": 89.6,
      "total_bytes": 34113015808,
      "total_gb": 31.77,
      "total_mb": 32532.71
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
        "link_generation": "Gen2",
        "manufacturer": "NVIDIA",
        "max_lane_width": "x16",
        "max_link_generation": "Gen4",
        "memory_total_bytes": 25769803776,
        "name": "NVIDIA GeForce RTX 3090",
        "revision": "0xa1",
        "status": "OK",
        "subsystem_device_id": "0x1454",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      }
    ],
    "motherboard": {
      "chipset": "INTEL Intel(R) SMBus - 7A23",
      "manufacturer": "Micro-Star International Co., Ltd.",
      "model_name": "MAG B760 TOMAHAWK WIFI (MS-7D96)",
      "pcie": {
        "max_lane_width": "x16",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe"
      },
      "version": "2.0"
    },
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
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
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
      "model_name": "INTEL(R) XEON(R) GOLD 6542Y",
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 322689507328,
      "available_gb": 300.53,
      "available_mb": 307740.7,
      "total_bytes": 405389791232,
      "total_gb": 377.55,
      "total_mb": 386609.83
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
        "max_lane_width": "x16",
        "max_link_generation": "Gen5",
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
        "link_generation": "Gen1",
        "manufacturer": "NVIDIA Corporation",
        "max_lane_width": "x16",
        "max_link_generation": "Gen5",
        "memory_total_bytes": 102641958912,
        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "revision": "0xa1",
        "subsystem_device_id": "0x204b",
        "subsystem_vendor_id": "0x10de",
        "vendor_id": "0x10de"
      }
    ],
    "motherboard": {
      "chipset": "ASPEED Technology, Inc. AST1150 PCI-to-PCI Bridge",
      "manufacturer": "Supermicro",
      "model_name": "X13DEG-QT",
      "pcie": {
        "max_lane_width": "x16",
        "max_link_generation": "Gen5",
        "max_link_speed": "32.0 GT/s PCIe"
      },
      "version": "1.10"
    },
    "npus": [
      {
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
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "Aries",
        "node_name": "aries0",
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
      "model_name": "13th Gen Intel(R) Core(TM) i5-13600K",
      "vendor": "GenuineIntel"
    },
    "dram": {
      "available_bytes": 27281420288,
      "available_gb": 25.41,
      "available_mb": 26017.59,
      "total_bytes": 33379606528,
      "total_gb": 31.09,
      "total_mb": 31833.27
    },
    "motherboard": {
      "chipset": "Intel Corporation Raptor Lake PCI Express Root Port #25",
      "manufacturer": "ASUSTeK COMPUTER INC.",
      "model_name": "ROG STRIX B760-I GAMING WIFI",
      "pcie": {
        "max_lane_width": "x16",
        "max_link_generation": "Gen5",
        "max_link_speed": "32.0 GT/s PCIe"
      },
      "version": "Rev 1.xx"
    },
    "npus": [
      {
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
        "manufacturer": "MOBILINT, Inc.",
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "MOBILINT NPU Accelerator",
        "node_name": "aries0",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
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
        "manufacturer": "MOBILINT, Inc.",
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "MOBILINT NPU Accelerator",
        "node_name": "aries1",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
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
        "manufacturer": "MOBILINT, Inc.",
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "MOBILINT NPU Accelerator",
        "node_name": "aries2",
        "product": "Aries",
        "revision": "0x2",
        "subsystem_device_id": "0x108B",
        "subsystem_vendor_id": "0x402",
        "vendor_id": "0x209F"
      },
      {
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
        "manufacturer": "MOBILINT, Inc.",
        "max_lane_width": "x8",
        "max_link_generation": "Gen4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "8",
        "memory_total_bytes": 17179869184,
        "name": "MOBILINT NPU Accelerator",
        "node_name": "aries3",
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
      "kernel_version": "6.17.0-23-generic",
      "name": "Linux",
      "version": "Ubuntu 24.04.4 LTS"
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
| **Power (W)** | ✅ (RAPL) | ✅ (RAPL DRAM) | ✅ (NVML) | ✅ (`mbltml`) |
| **Utilization (%)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mbltml`) |
| **Memory (MB/%)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mbltml`) |
| **Temperature (C)** | ✅ (`psutil`) | N/A | ✅ (NVML) | ✅ (`mbltml`) |
| **Static Info** | ✅ Host/OS/DRAM/Motherboard | ✅ Host/OS/DRAM/Motherboard | ✅ NVML + PCIe | ✅ PCIe + `mbltml` |
| **Per-Device Stats** | ✅ (Sockets) | ✅ (Sockets) | ✅ (GPU Indices) | ✅ (`mbltml` Device Indices) |

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
- **Static Info**: Reports CPU architecture, model, vendor, DRAM capacity/type/speed, motherboard metadata, OS details, and CPU power policy when available.

### NVIDIA GPU

Uses **NVML** (via `nvidia-ml-py`) for high-fidelity hardware monitoring.

- **Features**: Tracks total system GPU usage or specific indices (e.g., `GPUDeviceTracker(gpu_id=[0, 1])`).
- **Dependencies**: Requires NVIDIA Drivers and NVML library installed.
- **Temperature**: Reads on-die GPU temperature through NVML.
- **Static Info**: `GPUDeviceTracker.get_static_info()` reports the detected GPU count, tracked device names, NVIDIA driver version, and the raw NVML CUDA driver version. The `mblt-tracker collect` CLI provides richer NVML-discovered GPU metadata; for NVIDIA GPUs, PCIe link generation/width fields come from NVML, while OS PCIe discovery may add non-link identifiers and descriptive metadata.

### Host DRAM

Uses the **Intel RAPL DRAM domain** through `pyRAPL` for host DRAM power measurements.

- **Platform/Hardware**: Requires a host that exposes RAPL DRAM energy counters.
- **Permission**: Requires read access to Intel RAPL sysfs, similar to CPU power tracking.
- **Features**: Tracks all detected CPU socket DRAM domains by default, or specific socket IDs with `DRAMDeviceTracker(socket_id=0)` / `DRAMDeviceTracker(socket_id=[0, 1])`.
- **Metrics**: Reports total host DRAM power through standard keys (`avg_power_w`, `p99_power_w`, `max_power_w`) and DRAM-specific aliases (`avg_dram_power_w`, `p99_dram_power_w`, `max_dram_power_w`). Per-socket statistics are returned under `metrics["dram"]`.
- **Trace**: `DRAMDeviceTracker.get_trace()` returns total host DRAM power as `list[(timestamp, power_w)]`.
- **Static Info**: `DRAMDeviceTracker.get_static_info()` returns the same privacy-first host CPU, aggregate DRAM capacity with MB/GB display units, optional DIMM capacity/type/speed summaries, motherboard metadata, and OS metadata as host static collection. Individual DIMM identifiers are not exposed.

### Mobilint NPU

Uses **mbltml** for OS-independent Mobilint NPU telemetry on Linux and Windows.

- **Platform**: Supports Linux and Windows when the Mobilint driver and `mbltml` runtime library are available.
- **Device Selection**: Tracks all detected physical `mbltml` device indices by default, or selected indices with `NPUDeviceTracker(npu_id=0)` / `NPUDeviceTracker(npu_id=[0, 1])`.
- **Default Rail Policy**: The default `rail_metrics="npu"` mode reads total device power, NPU rail power/current/voltage, total utilization, memory usage, and temperature without changing the firmware rail selection register. This keeps the default monitoring path low-latency.
- **Shared Rail Register Limitation**: NPU, DDR, PMIC, and GoldFinger rail power/current/voltage readings share the same firmware register mapping. The register initially points to the NPU rail. Reading non-NPU rails requires changing the selected rail with `mbltmlSetExtraPmicID()`, and firmware may take up to about 1 second to refresh the register value.
- **Extra Rail Monitoring**: Set `rail_metrics="all"` or a list such as `rail_metrics=["npu", "ddr"]` to opt into DDR/PMIC/GoldFinger rail telemetry. Extra rails are sampled by a non-blocking state machine, so their effective sampling rate can be lower than `interval`; samples are recorded only after the firmware refresh delay has elapsed.
- **Metric Naming**: Rail-specific measurements use explicit names such as `avg_npu_rail_power_w`, `avg_ddr_rail_current_a`, and `avg_goldfinger_rail_voltage_v` to distinguish PMIC rail telemetry from total device power.
- **Static Info**: Reports Mobilint PCIe metadata and `mbltml` driver, firmware, device type, hardware version, and memory metadata where available.

---

## 📝 Metric Output Format

Calling `get_metric()` returns a dictionary with standardized cross-device keys where applicable. Missing or unavailable measurements are returned as `None`. Trackers may also include device-specific aliases or detailed telemetry keys; for example, NPU total telemetry keeps explicit `total_*` / `memory_usage_*` names while also exposing the standardized keys below for compatibility with CPU/GPU consumers.

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
  "samples": 100
}
```

Tracker-specific fields may also be present:

- **CPU**: `cpu` contains per-socket statistics keyed by socket ID.
- **GPU**: `gpu` contains per-GPU statistics keyed by GPU index. GPU-specific summary keys include `avg_gpu_util_pct`, `p99_gpu_util_pct`, `max_gpu_util_pct`, `avg_mem_util_pct`, and `p99_mem_util_pct`.
- **DRAM**: DRAM-specific power keys include `avg_dram_power_w`, `p99_dram_power_w`, and `max_dram_power_w`. `dram` contains per-socket statistics keyed by socket ID.
- **NPU**: NPU-specific keys include total device telemetry (`avg_total_power_w`, `avg_total_current_a`, `avg_total_voltage_v`), utilization (`avg_total_utilization_pct`), memory (`avg_memory_usage_mb`, `memory_total_mb`, `avg_memory_usage_pct`), temperature, and rail-specific telemetry (`avg_npu_rail_power_w`, `avg_ddr_rail_power_w`, `avg_pmic_rail_power_w`, `avg_goldfinger_rail_power_w`, plus matching current/voltage keys when available). For compatibility with CPU/GPU consumers, NPU total telemetry is also exposed through standardized aliases such as `avg_power_w`, `avg_utilization_pct`, `avg_memory_used_mb`, and `total_memory_mb`. `devices` contains per-`mbltml` device statistics keyed by device index. `rail_metrics` documents the selected rails and the 1-second firmware refresh limitation for extra rails.

### Time-Series Trace APIs

All trackers expose trace APIs for post-processing and plotting:

```python
tracker.get_trace()       # Power trace: list[(timestamp, power_w)]
tracker.get_util_trace()  # Utilization trace: list[(timestamp, utilization_pct)]
tracker.get_temp_trace()  # Temperature trace: list[(timestamp, temperature_c)]
```

NPU trackers additionally expose rail-specific power traces:

```python
tracker.get_npu_rail_power_trace()         # NPU rail power
tracker.get_ddr_rail_power_trace()         # DDR rail power, when enabled
tracker.get_pmic_rail_power_trace()        # PMIC rail power, when enabled
tracker.get_goldfinger_rail_power_trace()  # GoldFinger rail power, when enabled
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

- `hardware.cpu`: CPU architecture, model name, and vendor
- `hardware.dram`: total and available memory in bytes plus `total_mb`, `total_gb`, `available_mb`, and `available_gb`; optional privacy-safe aggregate fields include `ram_type`, `speed_mhz`, `configured_speed_mhz`, `module_count`, `modules`, and `theoretical_bandwidth_gbps` when available. Per-module entries may include `capacity_bytes`, `capacity_mb`, `capacity_gb`, DDR type, speed, width, and estimated bandwidth. Individual DIMM serial numbers, part numbers, and PCIe/device instance identifiers are not exposed.
- `hardware.motherboard`: optional baseboard `manufacturer`, `model_name`, `version`, best-effort `chipset`, and `pcie` support/capability summary. Motherboard serial numbers, asset tags, PCI bus addresses, and Windows instance IDs are not exposed.
- `hardware.motherboard.pcie`: optional maximum PCIe generation/speed/lane-width summary.
- `inference.os`: OS name, version, and kernel version
- `inference.cpu`: OS-independent CPU power policy object. Linux fills `governor`; Windows fills `power_plan`, `min_processor_state_pct`, and `max_processor_state_pct`. Unavailable OS-specific attributes are kept as `null`.
- `hardware.gpu`: `GPUDeviceTracker.get_static_info()` output with `device_count` and a `devices` list containing tracked GPU indices and names
- `hardware.gpus`: `mblt-tracker collect` output containing NVML-discovered NVIDIA GPU devices. For NVIDIA GPUs, current and maximum PCIe link generation/width fields are sourced from NVML; OS PCIe discovery may add non-link fields such as vendor/device IDs and descriptive metadata where available. Private PCIe instance identifiers such as `bus_address` and `pnp_device_id` are omitted from public output.
- `hardware.npus`: Mobilint PCIe devices, including vendor/device IDs, current/maximum link information, physical `mbltml` device index (`dev_no`), node name, device type, hardware version, memory metadata, and firmware metadata where available. Private PCIe instance identifiers such as `bus_address` and `pnp_device_id` are omitted from public output.
- `inference.gpu`: NVIDIA driver and CUDA driver versions. The CLI normalizes the CUDA driver version as a string such as `"13.0"`; `GPUDeviceTracker.get_static_info()` returns the raw NVML CUDA driver integer.
- `hardware.npus[].firmware`: per-NPU firmware metadata where available. Firmware metadata is collected through `mbltml` when available.
- `inference.npu_driver_version`: host Mobilint NPU driver version when available. Driver metadata is collected through `mbltml` when available.

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
