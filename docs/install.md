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

## Optional KTX2 Preview Decode

```bash
pip install 'fascat[ktx2]'
```

The `ktx2` extra installs the optional Python KTX2 decoder used by browser
preview validation when `KHR_texture_basisu` textures are present. It is
available on Python 3.11 and newer; without it, Fascat falls back to glTF
Transform plus KTX-Software when those external tools are installed.

## PyPI

Latest release: [pypi.org/project/fascat](https://pypi.org/project/fascat)

## Verify

```bash
fascat version
```
