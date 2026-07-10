from __future__ import annotations

import inspect

import fascat.runtime as runtime


def test_runtime_package_exports_public_api() -> None:
    expected = {
        "RuntimeBrowserOptions",
        "RuntimeBrowserRenderOptions",
        "RuntimeBrowserRenderReport",
        "RuntimeBrowserReport",
        "RuntimeEngineName",
        "RuntimeEngineOptions",
        "RuntimeEngineReport",
        "copy_engine_runtime_harness",
        "measure_browser_runtime",
        "measure_engine_runtime",
        "write_browser_render_preview",
    }

    assert set(runtime.__all__) == expected
    assert all(hasattr(runtime, name) for name in expected)


def test_runtime_package_preserves_public_function_signatures() -> None:
    assert str(inspect.signature(runtime.measure_browser_runtime)) == (
        "(path: 'str | Path', options: 'RuntimeBrowserOptions | None' = None) -> 'RuntimeBrowserReport'"
    )
    assert str(inspect.signature(runtime.write_browser_render_preview)) == (
        "(path: 'str | Path', preview_path: 'str | Path', "
        "options: 'RuntimeBrowserRenderOptions | None' = None) -> 'RuntimeBrowserRenderReport'"
    )
    assert str(inspect.signature(runtime.measure_engine_runtime)) == (
        "(path: 'str | Path', options: 'RuntimeEngineOptions') -> 'RuntimeEngineReport'"
    )
    assert str(inspect.signature(runtime.copy_engine_runtime_harness)) == (
        "(engine: 'RuntimeEngineName', destination: 'str | Path') -> 'Path'"
    )
