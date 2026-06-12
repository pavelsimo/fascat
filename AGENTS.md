# Agent Instructions — fascat

This file is the single canonical source of agent instructions for this repository.
`CLAUDE.md` is a symlink to this file. Do not create separate `CLAUDE.md` or `CODEX.md` variants.

## Project Structure

```
fascat/
├── fascat/          # Python package
│   ├── __init__.py           # version string
│   ├── __main__.py           # python -m fascat entry point
│   └── cli.py                # Typer app, global flags, subcommands
├── tests/                    # pytest test suite
├── docs/                     # Markdown documentation source
├── scripts/                  # Build tooling (docs site builder, etc.)
├── .github/workflows/        # CI, release, and pages workflows
├── pyproject.toml            # project metadata, deps, tool config
├── Makefile                  # All developer tasks
└── .lefthook.yml             # Git hook configuration
```

## Build / Test / Dev Commands

```bash
make install      # uv sync --dev (install project + dev deps)
make build        # uv build (wheel + sdist to dist/)
make test         # run pytest with coverage
make coverage     # open HTML coverage report in browser
make lint         # ruff check + mypy
make fmt          # ruff format + ruff check --fix
make fmt-check    # CI-safe format + lint check (exits non-zero if dirty)
make ci           # full gate: fmt-check + lint + test + build
make docs         # build documentation site to dist/docs-site/
make tools        # install development tools (lefthook)
```

## Coding Style

- All errors written to stderr; primary data to stdout
- JSON output and JSON error payloads are primary output and must go to stdout
- Use `err.print(...)` (Rich Console on stderr) for diagnostics; `out.print(...)` for data
- Respect `NO_COLOR`, `TERM=dumb`, non-TTY streams, and `--no-color` everywhere color is used
- Prompts only when `sys.stdin.isatty()` is True; `--no-input` disables all prompts
- Use `--dry-run` / `-n` before any state-changing operation
- Keep global flags valid before or after subcommands; update the `run()` normalizer when adding global flags
- File path arguments should accept `-` for stdin/stdout where meaningful
- `-h` / `--help` and `-V` / `--version` should work anywhere in the invocation and ignore other arguments
- Flag names are lowercase hyphenated; short flags only for the most common (`-v`, `-q`, `-n`, `-V`)
- Subcommands live as `@app.command()` functions in `fascat/cli.py`
  or in separate modules imported and registered there
- Type annotations on all public functions; mypy strict mode is enforced

## Testing Guidelines

- Tests live in `tests/`, one file per module under test
- Use `typer.testing.CliRunner` for CLI integration tests
- Use `pytest.mark.parametrize` for table-driven tests
- Aim for ≥ 70% coverage on `fascat/`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_cli.py::test_version_flag -v`

## Commit & PR Guidelines

Commits follow the [/commit skill](https://github.com/pavelsimo/commit) convention:
`<emoji> <lowercase imperative summary>`

The emoji carries the type signal — no `feat:` or `fix:` prefix. Body (after a blank line)
explains *why*, not how. No trailing period. All lowercase.

Common emoji:
- ✨ new feature · 🐛 bug fix · ♻️ refactor · 📝 docs
- 👷 CI/CD · 🔧 config · ⬆️ dependency bump · 🔥 remove code

PR description must include:
- What changed and why
- How it was tested
- Any new flags or breaking changes
