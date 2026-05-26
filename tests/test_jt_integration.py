from __future__ import annotations

import os
from pathlib import Path

import pytest

import fascat as fc


@pytest.mark.requires_jt
def test_native_jt_sample_imports() -> None:
    if not fc.has_native_jt_backend():
        pytest.skip("native Open Cascade JT bindings are not installed")
    sample = os.environ.get("FASCAT_JT_SAMPLE")
    if sample is None:
        pytest.skip("set FASCAT_JT_SAMPLE to a license-clean .jt file")

    asset = fc.read_jt(Path(sample))

    assert asset.root.name
    assert asset.report.steps[0].name == "import"
    assert asset.report.steps[0].options["format"] == "JT"
    assert asset.report.steps[0].options["backend"] == "OCP.JTCAFControl"
    assert asset.report.input_stats == asset.stats()
