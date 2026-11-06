# CHANGELOG

## 1.1.0

### Added

- Added `JetsonDeviceTracker` for NVIDIA Jetson. Jetson GPUs report no power or
  utilization through NVML, so the tracker reads the on-module INA3221 rails
  from hwmon sysfs (no root required) and reports module power through the
  standard `*_power_w` keys plus per-rail statistics for every rail under
  `rails`. Total power excludes rails that another rail already includes: it
  is `VDD_IN` alone when present (Orin NX/Nano), and on AGX Orin it is
  `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0` without the `VDDQ_VDD2_1V8AO`
  sub-rail. Explicit `rails=[...]` are summed as requested. This is on-module
  power only; carrier-board loads such as the fan, NVMe, USB, Ethernet, and
  PCIe cards are not measured. The tracker also records integrated GPU load,
  the `gpu-thermal` zone temperature, and the system memory shared by the CPU
  and GPU. `JetsonDeviceTracker.is_available()` detects a Jetson with readable
  rails, and `get_static_info()` reports the device-tree model, L4T release,
  all rail labels (`power_rails`), and the rails summed into total power
  (`power_total_rails`).

### Fixed

- Made `stop()` a no-op on a tracker that is not running. A repeated `stop()`
  previously took one more sample, so callers that stop a phase tracker and
  stop it again in cleanup code appended a late sample to an already closed
  window and over-counted that window's duration and energy. This affected
  every backend; for example, a prefill tracker stopped again after decode
  reported energy for the whole request.
- Prevented optional sudo credentials used for Linux DMI collection from being
  exposed to executables substituted through an attacker-controlled `PATH`. The
  password-bearing `sudo -S` call now runs only a `sudo` found at a trusted
  absolute path and is skipped otherwise; credential-free `dmidecode` and
  `sudo -n` attempts still fall back to a `PATH` lookup so hosts with
  non-standard layouts keep their DIMM and motherboard coverage.

## 1.0.0

### Changed

- Treat `mbltml` as the source of truth for Mobilint NPU static device entries;
  PCIe discovery now enriches those entries only and is no longer used as an NPU
  fallback discovery or filter path.
- Removed CLI NPU PCIe filter options (`--pcie-vendor-id`, `--pcie-device-id`,
  and `--pcie-class-filter`) and the corresponding `NPUDeviceTracker` PCIe
  filter environment behavior.
- Switched Mobilint NPU tracking from `mobilint-cli` subprocess parsing to the
  OS-independent `mbltml` Python API.
- Made `mbltml` a required runtime dependency because `mblt-tracker` is built as
  a Mobilint NPU-oriented tracker.
- Updated the supported Python range to `>=3.10,<3.13`.
- Changed `NPUDeviceTracker` device selection to use physical `mbltml` device
  indices instead of the previous `mobilint-cli` logical card grouping.
- Removed the `status_cmd` constructor argument and the legacy
  `device_tracker_npu.sh` sampling helper.
- Added NPU metric names based on total device metrics and extra PMIC rail
  metrics, with CPU/GPU-compatible generic aliases such as `avg_power_w`,
  `avg_utilization_pct`, and `total_memory_mb`.
- Added optional rail selection through `rail_metrics`. The default collects the
  NPU rail without changing the firmware rail selection; DDR, PMIC, and
  GoldFinger rails can be requested explicitly and respect the firmware refresh
  delay.
- Moved NPU driver, firmware, PCIe, and memory static metadata collection to the
  `mbltml` path for both Windows and Linux.

### Fixed

- Replaced privileged-container CPU telemetry guidance with a least-privilege
  setup: a udev rule that grants a dedicated `powercap` group read access to
  the RAPL `energy_uj` counters, and a Docker example that bind-mounts the
  backing `/sys/devices/virtual/powercap` hierarchy read-only at
  `/sys/class/powercap`, runs as a non-root user with that group added via
  `--group-add`, and adds no capabilities.
- Avoided over-merging filtered NPU static metadata by no longer matching
  Mobilint devices on `vendor_id` alone.
- Preserved public static output sanitization while merging `mbltml` NPU
  metadata into PCIe-discovered device entries.
