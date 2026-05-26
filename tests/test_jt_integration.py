from __future__ import annotations

import os
from pathlib import Path

import pytest

import fascat as fc


def _sample_paths() -> list[Path]:
    sample_dir = os.environ.get("FASCAT_JT_SAMPLE_DIR")
    if sample_dir is not None:
        paths = sorted(Path(sample_dir).glob("*.jt"))
        if not paths:
            pytest.fail(f"no .jt files found in FASCAT_JT_SAMPLE_DIR: {sample_dir}")
        return paths

    sample = os.environ.get("FASCAT_JT_SAMPLE")
    if sample is not None:
        return [Path(sample)]

    pytest.skip("set FASCAT_JT_SAMPLE or FASCAT_JT_SAMPLE_DIR to license-clean .jt input")


@pytest.mark.requires_jt
def test_native_jt_sample_imports() -> None:
    if not fc.has_native_jt_backend():
        pytest.skip("native Open Cascade JT bindings are not installed")

    for sample in _sample_paths():
        if not sample.is_file():
            pytest.fail(f"JT sample does not exist: {sample}")

        asset = fc.read_jt(sample)

        assert asset.root.name
        assert asset.report.steps[0].name == "import"
        assert asset.report.steps[0].options["format"] == "JT"
        assert asset.report.steps[0].options["backend"] == "OCP.JTCAFControl"
        assert asset.report.input_stats == asset.stats()
