# Repository Agent Guide

## Purpose

Maintain `mblt-tracker`, the Python library and CLI that collect dynamic CPU,
DRAM, NVIDIA GPU, and Mobilint NPU metrics plus privacy-safe static metadata.

`AGENTS.md` and `CLAUDE.md` are byte-for-byte mirrors. The repository skill is
also mirrored under `.agents/skills` and `.claude/skills`. Update both copies in
the same change and run the synchronization checks before finishing.

## Repository Map

- `mblt_tracker/device_tracker.py`: shared tracker lifecycle and scheduler.
- `mblt_tracker/device_tracker_cpu.py`: Intel RAPL package power plus `psutil`
  utilization, host memory, and best-effort temperature.
- `mblt_tracker/device_tracker_dram.py`: Intel RAPL DRAM-domain power.
- `mblt_tracker/device_tracker_gpu.py`: NVIDIA NVML discovery and telemetry.
- `mblt_tracker/device_tracker_npu.py`: `mbltml` discovery, telemetry, rail
  state machine, and NPU static metadata.
- `mblt_tracker/static_info.py`: Linux/Windows host, OS, DRAM, motherboard,
  PCIe, NVML, driver, firmware, merge, schema-cleaning, and sanitization logic.
- `mblt_tracker/cli.py`: `mblt-tracker collect` orchestration and JSON output.
- `mblt_tracker/_types.py`: public static-information `TypedDict` schemas.
- `mblt_tracker/__init__.py`: package version and public tracker exports.
- `tests/`: mocked unit/regression tests and hardware integration workloads.
- `pyproject.toml`: package metadata, dependencies, entry point, and tool config.
- `.github/workflows/ci.yml`: Python 3.10-3.12 package and test matrix.
- `.github/workflows/publish.yml`: release build, TestPyPI, and PyPI publication.
- `.github/workflows/check-agent-guides.yml`: mirrored-guide CI guard.

## Public Tracker Lifecycle

All public trackers derive from `BaseDeviceTracker`. Construction validates the
interval and backend-specific device selection. `start()` resets prior data,
takes one best-effort sample, starts a background scheduler, and installs one
interval job. `stop()` takes a final best-effort sample and shuts the scheduler
down. Backends implement summaries, traces, static information, and reset.

Preserve this lifecycle and keep repeated `start()` and `stop()` safe. Sampling
failures should not terminate the workload when a measurement can be reported
as unavailable. Return copies of trace lists, not internal mutable storage.

## Backend Responsibilities and Metrics

- CPU uses pyRAPL package energy for power and `psutil` for utilization, host
  memory, and temperature. Power is aggregated across selected sockets.
- DRAM uses the pyRAPL DRAM energy domain, reports aggregate and per-socket
  power, and fails clearly at construction when that domain is unsupported.
- GPU uses NVML indices and reports aggregate and per-device power,
  utilization, memory, and temperature. Unavailable power or temperature must
  not discard otherwise valid samples from the same tick.
- NPU uses physical `mbltml` device indices for discovery, selection,
  per-device metrics, and static identity. Never renumber a selected subset.
  Total device telemetry is separate from extra-PMIC rail telemetry.

Keep standardized aggregate aliases stable across backends:
`*_power_w`, `*_utilization_pct`, `*_memory_used_mb`,
`*_memory_used_pct`, `total_memory_mb`, and `*_temperature_c`, with
`avg_`, `p99_`, and `max_` statistics where supported. Preserve documented
backend-specific keys and aliases. Represent a missing measurement with
`None`, an empty trace, or a zero sample count; do not fabricate a number or
suppress unrelated measurements.

## NPU Rail Invariants

Extra NPU, DDR, PMIC, and GoldFinger rails share a firmware register. Selecting
a rail with `mbltmlSetExtraPmicID()` does not make its values current
immediately:

- explicitly select even the default NPU rail instead of trusting prior state;
- wait the firmware refresh period before recording a selected rail;
- cycle selected rails without blocking the scheduler thread;
- keep total device telemetry independent of rail availability;
- skip unsupported rails without breaking supported ones;
- restore NPU rail selection for every selected physical device on stop, even
  when shutdown or sampling raises.

## Static Information and CLI

`mblt-tracker collect` obtains host data and one OS PCIe snapshot, merges NVML
GPU metadata and `mbltml` NPU metadata, enriches those authoritative devices
with matching OS PCIe fields, sanitizes the merged result, applies the
`TypedDict` schema, and writes sorted JSON to stdout or `--output`.

NVML is authoritative for NVIDIA GPU identity and link data. `mbltml` is
authoritative for NPU discovery, physical `dev_no`, identity, driver, and
firmware. PCIe enumeration only enriches NPU entries; it must never create an
NPU that `mbltml` did not discover. Reuse a captured PCIe list so one collection
is internally consistent.

Public static output must remain privacy-safe, including with
`--all-pcie-devices`. Do not expose DIMM part or serial numbers, motherboard
serial or asset tags, PCIe bus addresses, or Windows PnP device instance IDs.
Collect private identifiers internally only for matching, then remove them
recursively before schema cleaning or JSON output. Static collection is best
effort: omit unavailable optional fields while preserving the stable
`hardware` and `inference` structure.

## Packaging and Releases

Python support, runtime and development dependencies, the console entry point,
and Ruff/mypy/pytest settings live in `pyproject.toml`. The runtime version
comes from `mblt_tracker.__version__`; align it with release intent and
user-visible changelog entries.

Publishing is release-driven. The publish workflow builds distributions once,
publishes the same artifact to TestPyPI, then publishes it to PyPI through
trusted environments. Do not add long-lived publishing credentials or bypass
the TestPyPI gate.

## Tests

Mock backend libraries, commands, clocks, and platform APIs for
hardware-independent unit tests. These tests must run on ordinary CI hosts and
unit-test failures are blocking.

`tests/test_gpu.py` and `tests/test_npu.py` are hardware-dependent workloads;
their module-level skips are expected when CUDA, an NPU, a runtime, or a model
dependency is unavailable. CPU/DRAM capabilities may also be absent on some
hosts. Keep skip reasons explicit, distinguish an unavailable environment from
a regression, and never weaken portable unit coverage because hardware is
missing.

## Documentation Maintenance

When a remarkable change affects public APIs, metrics, backends, dependencies,
supported Python versions or platforms, privacy behavior, CLI/static JSON,
typed schemas, repository structure, packaging or releases, tests, CI, or
validation commands, update in the same change:

1. the implementation and focused tests;
2. affected `README.md` sections and `CHANGELOG.md` when user-visible;
3. `AGENTS.md` and `CLAUDE.md`;
4. both `maintain-mblt-tracker` skill copies and their metadata when the skill
   scope changes.

Keep every Codex/Claude pair byte-identical. Never update only one mirror.

## Validation

Install and run the portable validation set:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m compileall -q mblt_tracker tests
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/maintain-mblt-tracker
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .claude/skills/maintain-mblt-tracker
cmp AGENTS.md CLAUDE.md
cmp .agents/skills/maintain-mblt-tracker/SKILL.md .claude/skills/maintain-mblt-tracker/SKILL.md
cmp .agents/skills/maintain-mblt-tracker/agents/openai.yaml .claude/skills/maintain-mblt-tracker/agents/openai.yaml
git diff --check
```

Parse changed workflow YAML and run focused tests while iterating. Run hardware
integration only where its devices and runtimes exist, and report skips
explicitly.

## Git Hygiene

- Preserve unrelated user changes and keep commits focused.
- Add regression tests for changed contracts and failure paths.
- Do not hide backend, privacy, packaging, or CI regressions to make checks pass.
- Review the final diff for schema drift, stale docs, and mirror drift.
