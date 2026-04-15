from __future__ import annotations

import json
import os
import subprocess


def test_npu_shell_parser_reads_temperature_from_status_output(
    monkeypatch, tmp_path
) -> None:
    cli_path = tmp_path / "mobilint-cli"
    cli_path.write_text(
        """#!/usr/bin/env bash
cat <<'EOF'
2026-04-15 16:06:59
+------------------------------------------------------------------------------------------+
| Mobilint-NPU-Monitor                           Drivers - Aries: 1.12.0  Regulus: N/A     |
+------------------------------------------------------------------------------------------+
| NPU  Name                     |   Pwr:NPU/Total |     Clock:NPU/Bus |       Memory-Usage |
| Sig  Temp    Firmware Version |   Cur:NPU/Total |                   |           NPU-Util |
|===============================+=================+===================+====================|
|   0  Aries(aries0)            |   2.11W   7.87W |   50MHz /  150MHz |      0MB / 16384MB |
|   0  49 C               1.2.4 |   0.17A   0.65A |                   |              0.00% |
+-------------------------------+-----------------+-------------------+--------------------+
EOF
""",
        encoding="utf-8",
    )
    cli_path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", "mblt_tracker/device_tracker_npu.sh", "--sample-once", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["npu_power_w"] == 2.11
    assert payload["total_power_w"] == 7.87
    assert payload["npu_util_pct"] == 0.0
    assert payload["npu_mem_used_mb"] == 0
    assert payload["npu_mem_total_mb"] == 16384
    assert payload["npu_temp_c"] == 49
