---
name: maintain-mblt-tracker
description: Maintain the mblt-tracker Python package and CLI. Use when changing tracker lifecycle, CPU/DRAM/GPU/NPU backends, metric names or aliases, NPU rail sampling, static metadata or privacy, CLI JSON, TypedDict schemas, dependencies, packaging, tests, CI, releases, or synchronized repository guides.
---

# Maintain MBLT Tracker

## Establish Scope

1. Read the root `AGENTS.md` or `CLAUDE.md` completely.
2. Inspect the public tracker, backend, static-info/CLI, schema, test, workflow,
   and README surfaces affected by the request.
3. Preserve unrelated worktree changes and existing public keys unless the
   change explicitly migrates them.

## Locate the Change Surface

- Change shared scheduling and lifecycle behavior in `device_tracker.py`.
- Change telemetry reads and summaries in the matching
  `device_tracker_{cpu,dram,gpu,npu}.py`.
- Change host, platform, PCIe, NVML, merge, or sanitization behavior in
  `static_info.py`.
- Change collection order and JSON output in `cli.py`.
- Change static output types and cleaning schemas in `_types.py`.
- Change exports or the runtime version in `__init__.py`.
- Change dependencies, Python support, entry points, or tool configuration in
  `pyproject.toml`.
- Add focused tests in the closest unit module; reserve hardware workload tests
  for behavior that requires real devices.

## Preserve Contracts

- Keep `start()` reset and immediate sampling, interval polling, final sampling,
  and safe scheduler shutdown consistent across trackers.
- Keep standardized power, utilization, memory, and temperature aggregate keys
  and documented backend-specific aliases synchronized.
- Represent unavailable measurements as `None`, empty traces, or zero samples;
  preserve valid measurements from the same tick.
- Treat NPU indices as physical `mbltml` device numbers. Do not renumber a
  selected subset.
- Explicitly select extra-PMIC rails, wait for the firmware refresh period,
  sample the rail cycle without blocking, and restore NPU selection on stop.
- Treat NVML and `mbltml` as authoritative for GPU and NPU discovery
  respectively. Use OS PCIe data only to enrich discovered NPU entries.
- Sanitize private DIMM, motherboard, PCIe bus, and Windows PnP identifiers
  after internal matching and before public static output.
- Keep best-effort static collection useful when optional commands, fields,
  hardware, or runtimes are unavailable.

## Update the Contract

1. Update implementation and focused regression tests together.
2. Update `README.md` and `CHANGELOG.md` for user-visible behavior.
3. Update both root guides for remarkable API, metric, backend, dependency,
   platform, privacy, CLI, schema, layout, packaging, test, CI, or validation
   changes.
4. Update both copies of this skill and their `agents/openai.yaml` metadata when
   its triggering scope changes.
5. Keep every Codex/Claude pair byte-identical.

## Validate

Run the narrowest focused tests while iterating, then run:

```bash
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

Mock hardware APIs, platform commands, and time for portable unit validation.
Run GPU/NPU integration workloads only when the required devices and runtimes
are present. Report skips as unavailable hardware validation, not as passing
device coverage, and treat portable unit failures as blocking.
