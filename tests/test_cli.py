from __future__ import annotations

import json

from mblt_tracker import cli


def test_collect_prints_static_info_as_json(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "pcie_vendor_id": None,
            "pcie_device_id": None,
            "pcie_class_filter": None,
        }
        return {"hardware.host.cpu.architecture": "x86_64"}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware.host.cpu.architecture": "x86_64"
    }


def test_collect_writes_static_info_to_output_file(monkeypatch, tmp_path, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "pcie_vendor_id": "1ed5",
            "pcie_device_id": "0100",
            "pcie_class_filter": "0x12",
        }
        return {"hardware.pcie.npu.vendor_id": "0x1ed5"}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    output = tmp_path / "nested" / "static_info.json"

    exit_code = cli.main(
        [
            "collect",
            "--output",
            str(output),
            "--pcie-vendor-id",
            "1ed5",
            "--pcie-device-id",
            "0100",
            "--pcie-class-filter",
            "0x12",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "hardware.pcie.npu.vendor_id": "0x1ed5"
    }