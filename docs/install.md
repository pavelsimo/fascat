---
title: Install
description: Install fascat on macOS, Linux, or Windows
---

## Homebrew (macOS / Linux)

```bash
brew tap pavelsimo/homebrew-tap
brew install fascat
```

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

Default installs include the Python KTX2 decoder used by browser preview
validation when `KHR_texture_basisu` textures are present on supported Python
3.11+ Linux/Windows x86_64 environments. The `ktx2` extra is retained for
explicit installs in compatible environments. On unsupported platforms, Fascat
falls back to glTF Transform plus KTX-Software when those external tools are
installed.

## PyPI

Latest release: [pypi.org/project/fascat](https://pypi.org/project/fascat)

## Verify

```bash
fascat version
```
