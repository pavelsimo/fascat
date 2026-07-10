from __future__ import annotations

import base64
import binascii
import json
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, unquote_to_bytes, urlparse
from urllib.request import url2pathname

import numpy as np
from numpy.typing import NDArray

from fascat import _subprocess
from fascat.io.gltf import validate_gltf

from .browser import _browser_command, _int, _parse_browser_payload
from .html import _runtime_browser_render_html
from .options import RuntimeBrowserRenderOptions, RuntimeBrowserRenderReport

_MAX_BROWSER_RENDER_SCREENSHOT_DATA_URI_LENGTH = 16 * 1024 * 1024


@dataclass(frozen=True)
class _BrowserRenderPreflight:
    required_extensions: tuple[str, ...] = ()
    unsupported_extensions: tuple[str, ...] = ()
    decoded_extensions: tuple[str, ...] = ()
    preview_limitations: tuple[str, ...] = ()
    draco_decode_required: bool = False
    meshopt_decode_required: bool = False
    texture_decode_required: bool = False
    fatal: bool = False


def write_browser_render_preview(
    path: str | Path,
    preview_path: str | Path,
    options: RuntimeBrowserRenderOptions | None = None,
) -> RuntimeBrowserRenderReport:
    opts = options or RuntimeBrowserRenderOptions()
    asset_path = Path(path)
    if asset_path.suffix.lower() not in {".gltf", ".glb"}:
        raise ValueError("browser render preview only supports glTF/GLB outputs")
    if not asset_path.exists():
        raise FileNotFoundError(f"missing runtime asset: {asset_path}")

    preflight = _browser_render_preflight(asset_path)
    output_path = Path(preview_path)
    with tempfile.TemporaryDirectory(prefix="fascat-browser-render-", ignore_cleanup_errors=True) as directory:
        workdir = Path(directory)
        render_asset_path = asset_path
        if preflight.draco_decode_required:
            try:
                render_asset_path = _write_draco_decoded_preview_asset(render_asset_path, workdir / "draco-decoded.glb")
            except Exception as exc:
                preflight = _preflight_draco_decode_failed(preflight, exc)
            else:
                preflight = _preflight_draco_decoded(preflight)
        if not preflight.fatal and preflight.meshopt_decode_required:
            try:
                render_asset_path = _write_meshopt_decoded_preview_asset(
                    render_asset_path, workdir / "meshopt-decoded.gltf"
                )
            except Exception as exc:
                preflight = _preflight_meshopt_decode_failed(preflight, exc)
            else:
                preflight = _preflight_meshopt_decoded(preflight)
        if not preflight.fatal and preflight.texture_decode_required:
            try:
                render_asset_path = _write_ktx2_decoded_preview_asset(render_asset_path, workdir / "ktx2-decoded.glb")
            except Exception as exc:
                preflight = _preflight_texture_decode_failed(preflight, exc)
            else:
                preflight = _preflight_texture_decoded(preflight)
        if preflight.fatal:
            try:
                validation_stats = validate_gltf(render_asset_path)
            except Exception:
                validation_stats = _gltf_document_stats(_read_gltf_json_document(asset_path))
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="unsupported",
                browser=None,
                error="; ".join(preflight.preview_limitations),
                preflight=preflight,
            )
        validation_stats = validate_gltf(render_asset_path)

        browser = _browser_command(opts)
        if browser is None:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="unavailable",
                browser=None,
                error="no chromium-compatible browser executable found",
                preflight=preflight,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        harness_path = Path(directory) / "browser-render.html"
        harness_path.write_text(_runtime_browser_render_html(render_asset_path.resolve(), opts), encoding="utf-8")
        command = _browser_render_report_invocation(browser, harness_path, opts)
        try:
            completed = _subprocess.run_guarded(command, timeout=opts.timeout_seconds)
        except subprocess.TimeoutExpired:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error="browser render preview timed out",
                preflight=preflight,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else f"browser exited with status {completed.returncode}"
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error=message,
                preflight=preflight,
            )
        payload = _parse_browser_payload(completed.stdout)
        if payload is None:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error="browser did not return render preview measurements",
                preflight=preflight,
            )
        status = str(payload.get("status", "failed"))
        error = payload.get("error")
        if status != "rendered":
            return RuntimeBrowserRenderReport(
                path=str(asset_path),
                status=status,
                browser=browser,
                preview_path=str(output_path),
                width=opts.width,
                height=opts.height,
                meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
                triangles=_int(payload.get("triangles"), validation_stats["triangles"]),
                textured_primitives=_int(payload.get("textured_primitives"), 0),
                sampled_textures=_int(payload.get("sampled_textures"), 0),
                quantized_primitives=_int(payload.get("quantized_primitives"), 0),
                required_extensions=preflight.required_extensions,
                unsupported_extensions=preflight.unsupported_extensions,
                decoded_extensions=preflight.decoded_extensions,
                preview_limitations=preflight.preview_limitations,
                error=str(error) if error is not None else None,
            )
        screenshot_data = payload.get("screenshot_data")
        if not isinstance(screenshot_data, str) or not _write_png_data_uri(screenshot_data, output_path):
            screenshot_command = _browser_render_screenshot_invocation(browser, harness_path, output_path, opts)
            try:
                screenshot_completed = _subprocess.run_guarded(screenshot_command, timeout=opts.timeout_seconds)
            except subprocess.TimeoutExpired:
                return _browser_render_report(
                    asset_path,
                    output_path,
                    opts,
                    validation_stats,
                    status="failed",
                    browser=browser,
                    error="browser render preview screenshot timed out",
                    preflight=preflight,
                )
            if screenshot_completed.returncode != 0:
                detail = (screenshot_completed.stderr or screenshot_completed.stdout).strip().splitlines()
                message = detail[-1] if detail else f"browser exited with status {screenshot_completed.returncode}"
                return _browser_render_report(
                    asset_path,
                    output_path,
                    opts,
                    validation_stats,
                    status="failed",
                    browser=browser,
                    error=message,
                    preflight=preflight,
                )
    if not output_path.exists():
        status = "failed"
        error = "browser did not write render preview screenshot"
    elif status == "rendered" and preflight.preview_limitations:
        status = "rendered_partial"
        error = "; ".join(preflight.preview_limitations)
    return RuntimeBrowserRenderReport(
        path=str(asset_path),
        status=status,
        browser=browser,
        preview_path=str(output_path),
        width=opts.width,
        height=opts.height,
        meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
        triangles=_int(payload.get("triangles"), validation_stats["triangles"]),
        textured_primitives=_int(payload.get("textured_primitives"), 0),
        sampled_textures=_int(payload.get("sampled_textures"), 0),
        quantized_primitives=_int(payload.get("quantized_primitives"), 0),
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=preflight.preview_limitations,
        error=str(error) if error is not None else None,
    )


def _browser_render_report_invocation(
    browser: str,
    harness_path: Path,
    options: RuntimeBrowserRenderOptions,
) -> list[str]:
    return [
        browser,
        "--headless=new",
        "--allow-file-access-from-files",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--use-gl=swiftshader",
        "--enable-unsafe-swiftshader",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={options.width},{options.height}",
        f"--virtual-time-budget={options.virtual_time_budget_ms}",
        "--dump-dom",
        harness_path.resolve().as_uri(),
    ]


def _browser_render_screenshot_invocation(
    browser: str,
    harness_path: Path,
    preview_path: Path,
    options: RuntimeBrowserRenderOptions,
) -> list[str]:
    return [
        browser,
        "--headless=new",
        "--allow-file-access-from-files",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--use-gl=swiftshader",
        "--enable-unsafe-swiftshader",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={options.width},{options.height}",
        f"--virtual-time-budget={options.virtual_time_budget_ms}",
        f"--screenshot={preview_path.resolve()}",
        harness_path.resolve().as_uri(),
    ]


def _write_png_data_uri(value: str, path: Path) -> bool:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        return False
    if len(value) > _MAX_BROWSER_RENDER_SCREENSHOT_DATA_URI_LENGTH:
        return False
    try:
        data = base64.b64decode(value[len(prefix) :], validate=True)
    except (binascii.Error, ValueError):
        return False
    path.write_bytes(data)
    return True


def _browser_render_report(
    asset_path: Path,
    preview_path: Path,
    options: RuntimeBrowserRenderOptions,
    validation_stats: dict[str, int],
    *,
    status: str,
    browser: str | None,
    error: str | None,
    preflight: _BrowserRenderPreflight | None = None,
) -> RuntimeBrowserRenderReport:
    checks = preflight or _BrowserRenderPreflight()
    return RuntimeBrowserRenderReport(
        path=str(asset_path),
        status=status,
        browser=browser,
        preview_path=str(preview_path),
        width=options.width,
        height=options.height,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        textured_primitives=0,
        sampled_textures=0,
        quantized_primitives=0,
        required_extensions=checks.required_extensions,
        unsupported_extensions=checks.unsupported_extensions,
        decoded_extensions=checks.decoded_extensions,
        preview_limitations=checks.preview_limitations,
        error=error,
    )


def _browser_render_preflight(asset_path: Path) -> _BrowserRenderPreflight:
    document = _read_gltf_json_document(asset_path)
    required = tuple(sorted(_string_list(document.get("extensionsRequired"))))
    unsupported_extensions: set[str] = set()
    limitations: list[str] = []

    draco_decode_required = _document_uses_draco(document)
    if draco_decode_required:
        limitations.append("browser preview requires Draco decode for KHR_draco_mesh_compression geometry")
    meshopt_decode_required = _document_has_meshopt_only_buffer_views(document, asset_path)
    if meshopt_decode_required:
        limitations.append("browser preview requires meshopt decode for bufferViews without fallback buffer data")
    texture_decode_required = _document_uses_basis_textures(document)
    if texture_decode_required:
        unsupported_extensions.add("KHR_texture_basisu")
        limitations.append("browser preview requires KTX2/Basis decode for texture sampling")

    return _BrowserRenderPreflight(
        required_extensions=required,
        unsupported_extensions=tuple(sorted(unsupported_extensions)),
        preview_limitations=tuple(limitations),
        draco_decode_required=draco_decode_required,
        meshopt_decode_required=meshopt_decode_required,
        texture_decode_required=texture_decode_required,
        fatal=False,
    )


def _preflight_draco_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "KHR_draco_mesh_compression"))),
        preview_limitations=_without_preview_limitations(preflight, "Draco", "KHR_draco"),
        draco_decode_required=False,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=preflight.texture_decode_required,
        fatal=preflight.fatal,
    )


def _preflight_draco_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    unsupported = tuple(sorted((*preflight.unsupported_extensions, "KHR_draco_mesh_compression")))
    limitations = _without_preview_limitations(preflight, "Draco", "KHR_draco") + (
        f"browser preview could not decode KHR_draco_mesh_compression: {exc}",
    )
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=unsupported,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=False,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=preflight.texture_decode_required,
        fatal=True,
    )


def _preflight_meshopt_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "EXT_meshopt_compression"))),
        preview_limitations=_without_preview_limitations(preflight, "meshopt", "EXT_meshopt"),
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=False,
        texture_decode_required=preflight.texture_decode_required,
        fatal=preflight.fatal,
    )


def _preflight_meshopt_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    unsupported = tuple(sorted((*preflight.unsupported_extensions, "EXT_meshopt_compression")))
    limitations = tuple(
        item for item in preflight.preview_limitations if "meshopt" not in item and "EXT_meshopt" not in item
    ) + (f"browser preview could not decode EXT_meshopt_compression: {exc}",)
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=unsupported,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=False,
        texture_decode_required=preflight.texture_decode_required,
        fatal=True,
    )


def _preflight_texture_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=_without_extensions(preflight.unsupported_extensions, "KHR_texture_basisu"),
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "KHR_texture_basisu"))),
        preview_limitations=_without_preview_limitations(preflight, "KTX2", "Basis", "KHR_texture_basisu"),
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=False,
        fatal=preflight.fatal,
    )


def _preflight_texture_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    limitations = _without_preview_limitations(preflight, "KTX2", "Basis", "KHR_texture_basisu") + (
        f"browser preview could not decode KHR_texture_basisu: {exc}",
        "browser preview renders fallback texture sources when present; otherwise geometry without KTX2/Basis texture sampling",
    )
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=tuple(sorted({*preflight.unsupported_extensions, "KHR_texture_basisu"})),
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=False,
        fatal=preflight.fatal,
    )


def _without_preview_limitations(preflight: _BrowserRenderPreflight, *patterns: str) -> tuple[str, ...]:
    return tuple(item for item in preflight.preview_limitations if not any(pattern in item for pattern in patterns))


def _without_extensions(extensions: tuple[str, ...], *names: str) -> tuple[str, ...]:
    return tuple(item for item in extensions if item not in names)


def _read_gltf_json_document(asset_path: Path) -> dict[str, Any]:
    if asset_path.suffix.lower() == ".glb":
        data = asset_path.read_bytes()
        if len(data) < 20 or data[0:4] != b"glTF":
            raise RuntimeError("GLB header is invalid")
        json_length = int.from_bytes(data[12:16], "little")
        json_type = int.from_bytes(data[16:20], "little")
        if json_type != 0x4E4F534A:
            raise RuntimeError("first GLB chunk is not JSON")
        end = 20 + json_length
        if end > len(data):
            raise RuntimeError("GLB JSON chunk length is invalid")
        loaded = json.loads(data[20:end].decode("utf-8"))
    else:
        loaded = json.loads(asset_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("glTF JSON document must be an object")
    return cast(dict[str, Any], loaded)


def _write_draco_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    _run_gltf_transform_copy(asset_path, output_path)
    decoded = _read_gltf_json_document(output_path)
    if _document_uses_draco(decoded):
        raise RuntimeError("glTF Transform copy preserved KHR_draco_mesh_compression")
    return output_path


def _run_gltf_transform_copy(input_path: Path, output_path: Path) -> None:
    command = [*_gltf_transform_runtime_command("copy"), "copy", str(input_path), str(output_path)]
    try:
        completed = _subprocess.run_guarded(command, timeout=_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"glTF Transform copy timed out after {_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"glTF Transform copy failed: {details}")


def _gltf_transform_runtime_command(operation: str) -> list[str]:
    from fascat.io.gltf import _gltf_transform_command

    try:
        return _gltf_transform_command()
    except RuntimeError as exc:
        raise RuntimeError(
            f"glTF Transform {operation} requires the glTF Transform CLI on PATH or FASCAT_GLTF_TRANSFORM"
        ) from exc


def _write_ktx2_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    python_error: Exception | None = None
    try:
        return _write_python_ktx2_decoded_preview_asset(asset_path, output_path.with_suffix(".gltf"))
    except Exception as exc:
        python_error = exc

    try:
        _run_gltf_transform_ktxdecompress(asset_path, output_path)
    except Exception as exc:
        if python_error is not None:
            raise RuntimeError(f"{python_error}; {exc}") from exc
        raise
    decoded = _read_gltf_json_document(output_path)
    if _document_uses_basis_textures(decoded):
        raise RuntimeError("glTF Transform ktxdecompress preserved KHR_texture_basisu")
    return output_path


def _write_python_ktx2_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    try:
        alktx2 = cast(Any, import_module("alktx2"))
    except ImportError as exc:
        raise RuntimeError("default alktx2 KTX2 decoder is not installed for this environment") from exc

    source_document, buffers = _read_gltf_json_and_buffers_for_preview(asset_path)
    document = _preview_document_copy(source_document, images=True, textures=True, extension_lists=True)
    images = document.get("images")
    if not isinstance(images, list):
        raise RuntimeError("glTF KTX2 decode requires an images array")
    basis_sources = _basis_texture_image_indices(document)
    decoded_images = 0
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            raise RuntimeError("glTF image must be an object")
        if image_index not in basis_sources and not _image_is_ktx2(image):
            continue
        ktx2_bytes = _load_gltf_image_bytes(image, document, buffers, asset_path.parent)
        decoded = alktx2.decode_ktx2_to_bytes(ktx2_bytes, format="png")
        if not isinstance(decoded, tuple) or len(decoded) != 2:
            raise RuntimeError("alktx2 decode_ktx2_to_bytes returned an invalid result")
        png_bytes, mime_type = decoded
        if not isinstance(png_bytes, bytes) or mime_type != "image/png":
            raise RuntimeError("alktx2 KTX2 decoder did not return PNG bytes")
        image["uri"] = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        image["mimeType"] = "image/png"
        image.pop("bufferView", None)
        decoded_images += 1

    if decoded_images == 0:
        raise RuntimeError("no KTX2/Basis images were decoded")

    _promote_basis_texture_sources(document)
    _remove_gltf_extension(document, "KHR_texture_basisu")
    _embed_preview_buffers(document, buffers)
    _rewrite_external_image_uris(document, asset_path.parent)
    if _document_uses_basis_textures(document):
        raise RuntimeError("alktx2 KTX2 decode preserved KHR_texture_basisu")
    output_path.write_text(json.dumps(document), encoding="utf-8")
    return output_path


def _run_gltf_transform_ktxdecompress(input_path: Path, output_path: Path) -> None:
    command = [*_gltf_transform_runtime_command("ktxdecompress"), "ktxdecompress", str(input_path), str(output_path)]
    try:
        completed = _subprocess.run_guarded(command, timeout=_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"glTF Transform ktxdecompress timed out after {_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"glTF Transform ktxdecompress failed: {details}")


def _write_meshopt_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    try:
        import meshoptimizer
    except ImportError as exc:
        raise RuntimeError("meshoptimizer is not installed") from exc

    source_document, buffers = _read_gltf_json_and_buffers_for_preview(asset_path)
    document = _preview_document_copy(source_document, images=True, buffer_views=True, extension_lists=True)
    buffer_views = document.get("bufferViews")
    if not isinstance(buffer_views, list):
        raise RuntimeError("glTF bufferViews must be an array")

    payload = bytearray()
    decoded_views = 0
    for view in buffer_views:
        if not isinstance(view, dict):
            raise RuntimeError("glTF bufferView must be an object")
        extensions = view.get("extensions")
        meshopt_extension = extensions.get("EXT_meshopt_compression") if isinstance(extensions, dict) else None
        if isinstance(meshopt_extension, dict):
            data = _decode_meshopt_buffer_view(meshopt_extension, buffers, meshoptimizer)
            decoded_views += 1
        else:
            data = _copy_buffer_view_bytes(view, buffers)
        offset = _append_preview_buffer(payload, data)
        view["buffer"] = 0
        view["byteOffset"] = offset
        view["byteLength"] = len(data)
        if isinstance(meshopt_extension, dict) and isinstance(meshopt_extension.get("byteStride"), int):
            if meshopt_extension.get("mode") == "ATTRIBUTES":
                view["byteStride"] = meshopt_extension["byteStride"]
            else:
                view.pop("byteStride", None)
        if isinstance(extensions, dict):
            extensions.pop("EXT_meshopt_compression", None)
            if not extensions:
                view.pop("extensions", None)

    if decoded_views == 0:
        raise RuntimeError("no meshopt-compressed bufferViews were found")
    document["buffers"] = [
        {
            "byteLength": len(payload),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(payload)).decode("ascii"),
        }
    ]
    _remove_gltf_extension(document, "EXT_meshopt_compression")
    _rewrite_external_image_uris(document, asset_path.parent)
    if _document_has_meshopt_only_buffer_views(document, output_path):
        raise RuntimeError("meshoptimizer decode preserved EXT_meshopt_compression")
    output_path.write_text(json.dumps(document), encoding="utf-8")
    return output_path


def _read_gltf_json_and_buffers_for_preview(asset_path: Path) -> tuple[dict[str, Any], list[bytes]]:
    binary_chunk = b""
    if asset_path.suffix.lower() == ".glb":
        document, binary_chunk = _read_glb_json_and_binary(asset_path)
    else:
        document = _read_gltf_json_document(asset_path)

    buffers: list[bytes] = []
    for index, buffer_value in enumerate(document.get("buffers", [])):
        if not isinstance(buffer_value, dict):
            raise RuntimeError(f"glTF buffer {index} must be an object")
        uri = buffer_value.get("uri")
        if isinstance(uri, str):
            buffers.append(_load_gltf_uri_bytes(uri, asset_path.parent))
        elif asset_path.suffix.lower() == ".glb" and index == 0:
            buffers.append(binary_chunk)
        else:
            buffers.append(b"")
    return document, buffers


def _preview_document_copy(
    source_document: dict[str, Any],
    *,
    images: bool = False,
    textures: bool = False,
    buffer_views: bool = False,
    extension_lists: bool = False,
) -> dict[str, Any]:
    document = dict(source_document)
    if images:
        _copy_preview_object_list(document, "images")
    if textures:
        _copy_preview_object_list(document, "textures", copy_extensions=True)
    if buffer_views:
        _copy_preview_object_list(document, "bufferViews", copy_extensions=True)
    if extension_lists:
        for field in ("extensionsUsed", "extensionsRequired"):
            value = source_document.get(field)
            if isinstance(value, list):
                document[field] = list(value)
    return document


def _copy_preview_object_list(document: dict[str, Any], field: str, *, copy_extensions: bool = False) -> None:
    value = document.get(field)
    if not isinstance(value, list):
        return
    copied: list[object] = []
    for item in value:
        if isinstance(item, dict):
            item_copy = dict(item)
            extensions = item_copy.get("extensions")
            if copy_extensions and isinstance(extensions, dict):
                item_copy["extensions"] = dict(extensions)
            copied.append(item_copy)
        else:
            copied.append(item)
    document[field] = copied


def _read_glb_json_and_binary(asset_path: Path) -> tuple[dict[str, Any], bytes]:
    data = asset_path.read_bytes()
    if len(data) < 20 or data[0:4] != b"glTF":
        raise RuntimeError("GLB header is invalid")
    json_document: dict[str, Any] | None = None
    binary_chunk = b""
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError("invalid GLB chunk header")
        chunk_length = int.from_bytes(data[offset : offset + 4], "little")
        chunk_type = int.from_bytes(data[offset + 4 : offset + 8], "little")
        offset += 8
        chunk = data[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise RuntimeError("invalid GLB chunk length")
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            loaded = json.loads(chunk.decode("utf-8").rstrip(" \x00"))
            if not isinstance(loaded, dict):
                raise RuntimeError("glTF JSON document must be an object")
            json_document = cast(dict[str, Any], loaded)
        elif chunk_type == 0x004E4942:
            binary_chunk = chunk
    if json_document is None:
        raise RuntimeError("GLB contains no JSON chunk")
    return json_document, binary_chunk


def _load_gltf_uri_bytes(uri: str, base_dir: Path) -> bytes:
    if uri.startswith("data:"):
        comma = uri.find(",")
        if comma < 0:
            raise RuntimeError("invalid data URI buffer")
        header = uri[:comma]
        payload = uri[comma + 1 :]
        if ";base64" in header:
            return base64.b64decode(payload)
        return unquote_to_bytes(payload)
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise RuntimeError(f"unsupported external glTF URI scheme: {parsed.scheme}")
    # url2pathname handles the /C:/... form a file URI produces on Windows.
    path = Path(url2pathname(parsed.path)) if parsed.scheme == "file" else base_dir / unquote(parsed.path)
    return path.read_bytes()


def _load_gltf_image_bytes(
    image: dict[str, Any],
    document: dict[str, Any],
    buffers: list[bytes],
    base_dir: Path,
) -> bytes:
    uri = image.get("uri")
    if isinstance(uri, str):
        return _load_gltf_uri_bytes(uri, base_dir)
    view_index = image.get("bufferView")
    if not isinstance(view_index, int):
        raise RuntimeError("glTF KTX2 image must use uri or bufferView")
    buffer_views = document.get("bufferViews")
    if not isinstance(buffer_views, list) or view_index < 0 or view_index >= len(buffer_views):
        raise RuntimeError("glTF KTX2 image references an invalid bufferView")
    view = buffer_views[view_index]
    if not isinstance(view, dict):
        raise RuntimeError("glTF KTX2 image bufferView must be an object")
    return _copy_buffer_view_bytes(view, buffers)


def _decode_meshopt_buffer_view(
    extension: dict[str, Any],
    buffers: list[bytes],
    meshoptimizer: Any,
) -> bytes:
    buffer_index = _required_int(extension, "buffer")
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise RuntimeError("meshopt extension references an invalid buffer")
    byte_offset = _optional_json_int(extension.get("byteOffset"), 0)
    byte_length = _required_int(extension, "byteLength")
    byte_stride = _required_int(extension, "byteStride")
    count = _required_int(extension, "count")
    mode = extension.get("mode")
    filter_name = str(extension.get("filter", "NONE"))
    if not isinstance(mode, str):
        raise RuntimeError("meshopt extension mode must be a string")
    source_buffer = buffers[buffer_index]
    source = source_buffer[byte_offset : byte_offset + byte_length]
    if len(source) != byte_length:
        raise RuntimeError("meshopt compressed bufferView is out of range")

    if mode == "ATTRIBUTES":
        decoded = meshoptimizer.decode_vertex_buffer(count, byte_stride, source)
        raw = decoded.view(np.uint8).reshape(-1)[: count * byte_stride].copy()
        raw = _apply_meshopt_filter(raw, count, byte_stride, filter_name, meshoptimizer)
        return raw.tobytes()
    if mode == "TRIANGLES":
        if filter_name != "NONE":
            raise RuntimeError("meshopt TRIANGLES mode does not support filters")
        decoded = meshoptimizer.decode_index_buffer(count, byte_stride, source)
        return cast(bytes, decoded.view(np.uint8).reshape(-1)[: count * byte_stride].tobytes())
    if mode == "INDICES":
        if filter_name != "NONE":
            raise RuntimeError("meshopt INDICES mode does not support filters")
        decoded = meshoptimizer.decode_index_sequence(count, byte_stride, source)
        return cast(bytes, decoded.view(np.uint8).reshape(-1)[: count * byte_stride].tobytes())
    raise RuntimeError(f"unsupported meshopt mode: {mode}")


def _apply_meshopt_filter(
    raw: NDArray[np.uint8],
    count: int,
    byte_stride: int,
    filter_name: str,
    meshoptimizer: Any,
) -> NDArray[np.uint8]:
    if filter_name == "NONE":
        return raw
    if filter_name == "OCTAHEDRAL":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_oct(raw, count, byte_stride))
    if filter_name == "QUATERNION":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_quat(raw, count, byte_stride))
    if filter_name == "EXPONENTIAL":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_exp(raw, count, byte_stride))
    raise RuntimeError(f"unsupported meshopt filter: {filter_name}")


def _copy_buffer_view_bytes(view: dict[str, Any], buffers: list[bytes]) -> bytes:
    buffer_index = _optional_json_int(view.get("buffer"), 0)
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise RuntimeError("glTF bufferView references an invalid buffer")
    byte_offset = _optional_json_int(view.get("byteOffset"), 0)
    byte_length = _required_int(view, "byteLength")
    data = buffers[buffer_index][byte_offset : byte_offset + byte_length]
    if len(data) != byte_length:
        raise RuntimeError("glTF bufferView is out of range")
    return data


def _append_preview_buffer(payload: bytearray, data: bytes) -> int:
    padding = (-len(payload)) % 4
    if padding:
        payload.extend(b"\x00" * padding)
    offset = len(payload)
    payload.extend(data)
    return offset


def _embed_preview_buffers(document: dict[str, Any], buffers: list[bytes]) -> None:
    document["buffers"] = [
        {
            "byteLength": len(data),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii"),
        }
        for data in buffers
    ]


def _remove_gltf_extension(document: dict[str, Any], extension_name: str) -> None:
    for field in ("extensionsUsed", "extensionsRequired"):
        value = document.get(field)
        if not isinstance(value, list):
            continue
        document[field] = [item for item in value if item != extension_name]
        if not document[field]:
            document.pop(field, None)


def _rewrite_external_image_uris(document: dict[str, Any], base_dir: Path) -> None:
    images = document.get("images")
    if not isinstance(images, list):
        return
    for image in images:
        if not isinstance(image, dict):
            continue
        uri = image.get("uri")
        if not isinstance(uri, str) or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        if parsed.scheme:
            continue
        image["uri"] = (base_dir / unquote(parsed.path)).resolve().as_uri()


def _gltf_document_stats(document: dict[str, Any]) -> dict[str, int]:
    meshes = document.get("meshes")
    accessors = document.get("accessors")
    if not isinstance(meshes, list) or not isinstance(accessors, list):
        return {"meshes": 0, "points": 0, "triangles": 0}
    points = 0
    triangles = 0
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        for primitive in primitives:
            if not isinstance(primitive, dict):
                continue
            attributes = primitive.get("attributes")
            if isinstance(attributes, dict) and isinstance(attributes.get("POSITION"), int):
                position_count = _accessor_count_from_json(accessors, attributes["POSITION"])
                points += position_count
            else:
                position_count = 0
            if isinstance(primitive.get("indices"), int):
                triangles += _accessor_count_from_json(accessors, primitive["indices"]) // 3
            elif position_count:
                triangles += position_count // 3
    return {"meshes": len(meshes), "points": points, "triangles": triangles}


def _accessor_count_from_json(accessors: list[object], index: int) -> int:
    if index < 0 or index >= len(accessors):
        return 0
    accessor = accessors[index]
    if not isinstance(accessor, dict) or not isinstance(accessor.get("count"), int):
        return 0
    return int(accessor["count"])


def _required_int(source: dict[str, Any], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"meshopt extension {key} must be an integer")
    return value


def _optional_json_int(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError("glTF integer value must be an integer")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _document_uses_draco(document: dict[str, Any]) -> bool:
    if "KHR_draco_mesh_compression" in _string_list(document.get("extensionsRequired")):
        return True
    for primitive in _iter_gltf_primitives(document):
        extensions = primitive.get("extensions")
        if isinstance(extensions, dict) and "KHR_draco_mesh_compression" in extensions:
            return True
    return False


def _document_has_meshopt_only_buffer_views(document: dict[str, Any], asset_path: Path) -> bool:
    buffer_views = document.get("bufferViews")
    accessors = document.get("accessors")
    if not isinstance(buffer_views, list) or not isinstance(accessors, list):
        return False
    referenced_buffer_views = _referenced_primitive_buffer_views(document, accessors)
    for index, view in enumerate(buffer_views):
        if index not in referenced_buffer_views:
            continue
        if not isinstance(view, dict):
            continue
        extensions = view.get("extensions")
        if not isinstance(extensions, dict) or "EXT_meshopt_compression" not in extensions:
            continue
        buffer_index = view.get("buffer")
        if not isinstance(buffer_index, int):
            return True
        if _buffer_is_meshopt_placeholder(document, buffer_index, asset_path):
            return True
    return False


def _buffer_is_meshopt_placeholder(document: dict[str, Any], buffer_index: int, asset_path: Path) -> bool:
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or buffer_index < 0 or buffer_index >= len(buffers):
        return True
    buffer = buffers[buffer_index]
    if not isinstance(buffer, dict):
        return True
    extensions = buffer.get("extensions")
    meshopt = extensions.get("EXT_meshopt_compression") if isinstance(extensions, dict) else None
    if isinstance(meshopt, dict) and meshopt.get("fallback") is True:
        return True
    uri = buffer.get("uri")
    if isinstance(uri, str):
        return False
    return not (asset_path.suffix.lower() == ".glb" and buffer_index == 0)


def _document_uses_basis_textures(document: dict[str, Any]) -> bool:
    if "KHR_texture_basisu" in _string_list(document.get("extensionsRequired")):
        return True
    textures = document.get("textures")
    if isinstance(textures, list):
        for texture in textures:
            if not isinstance(texture, dict):
                continue
            extensions = texture.get("extensions")
            if isinstance(extensions, dict) and "KHR_texture_basisu" in extensions:
                return True
    images = document.get("images")
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            mime_type = image.get("mimeType")
            uri = image.get("uri")
            if mime_type == "image/ktx2" or (isinstance(uri, str) and uri.lower().endswith(".ktx2")):
                return True
    return False


def _basis_texture_image_indices(document: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    textures = document.get("textures")
    if not isinstance(textures, list):
        return indices
    for texture in textures:
        if not isinstance(texture, dict):
            continue
        extensions = texture.get("extensions")
        basis = extensions.get("KHR_texture_basisu") if isinstance(extensions, dict) else None
        if isinstance(basis, dict) and isinstance(basis.get("source"), int):
            indices.add(cast(int, basis["source"]))
    return indices


def _promote_basis_texture_sources(document: dict[str, Any]) -> None:
    textures = document.get("textures")
    if not isinstance(textures, list):
        return
    for texture in textures:
        if not isinstance(texture, dict):
            continue
        extensions = texture.get("extensions")
        basis = extensions.get("KHR_texture_basisu") if isinstance(extensions, dict) else None
        if isinstance(basis, dict) and isinstance(basis.get("source"), int):
            texture["source"] = basis["source"]
        if isinstance(extensions, dict):
            extensions.pop("KHR_texture_basisu", None)
            if not extensions:
                texture.pop("extensions", None)


def _image_is_ktx2(image: dict[str, Any]) -> bool:
    mime_type = image.get("mimeType")
    uri = image.get("uri")
    return mime_type == "image/ktx2" or (isinstance(uri, str) and uri.lower().endswith(".ktx2"))


def _referenced_primitive_buffer_views(document: dict[str, Any], accessors: list[object]) -> set[int]:
    indices: set[int] = set()
    for accessor_index in _iter_primitive_accessor_indices(document):
        if accessor_index < 0 or accessor_index >= len(accessors):
            continue
        accessor = accessors[accessor_index]
        if not isinstance(accessor, dict):
            continue
        buffer_view = accessor.get("bufferView")
        if isinstance(buffer_view, int):
            indices.add(buffer_view)
    return indices


def _iter_primitive_accessor_indices(document: dict[str, Any]) -> Iterable[int]:
    for primitive in _iter_gltf_primitives(document):
        index = primitive.get("indices")
        if isinstance(index, int):
            yield index
        attributes = primitive.get("attributes")
        if not isinstance(attributes, dict):
            continue
        for attribute_index in attributes.values():
            if isinstance(attribute_index, int):
                yield attribute_index


def _iter_gltf_primitives(document: dict[str, Any]) -> Iterable[dict[str, Any]]:
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        return
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        for primitive in primitives:
            if isinstance(primitive, dict):
                yield primitive
