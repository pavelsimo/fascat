---
title: Install
description: Install fascat on macOS, Linux, or Windows
---

## pipx (recommended for isolated install)

```bash
pipx install fascat
```

## pip

```bash
pip install fascat
```

## KTX2 Preview Decode

```bash
pip install 'fascat[ktx2]'
```

The `ktx2` extra installs the Python KTX2 decoder used by browser preview
validation when `KHR_texture_basisu` textures are present; it supports Python
3.11+ on Linux/Windows x86_64. Base installs do not pull the decoder — without
the extra, Fascat falls back to glTF Transform plus KTX-Software when those
external tools are installed.

## glTF Compression Tooling

Draco export requires the glTF Transform CLI on `PATH`, or a
`FASCAT_GLTF_TRANSFORM` environment variable pointing at the executable.

KTX2/Basis texture export requires Node.js plus these packages installed in the
working directory, or in `FASCAT_NODE_MODULE_ROOT`:

```bash
npm install --save-dev @gltf-transform/core @gltf-transform/extensions ktx2-encoder sharp
```

## PyPI

Latest release: [pypi.org/project/fascat](https://pypi.org/project/fascat)

## Verify

```bash
fascat version
```
