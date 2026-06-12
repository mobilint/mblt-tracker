# CHANGELOG

## 1.0.0

### Changed

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

- Avoided over-merging filtered NPU static metadata by no longer matching
  Mobilint devices on `vendor_id` alone.
- Preserved public static output sanitization while merging `mbltml` NPU
  metadata into PCIe-discovered device entries.