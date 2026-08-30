# Contributing to xinterp

xinterp is a Rust extension module built with [maturin] and developed with
[uv]. uv owns the Python side (interpreter, virtual environment, dependencies,
test runs) and cargo owns the Rust side; the two are wired together so that
neither needs anything set by hand.

## Prerequisites

- [uv] — provisions the interpreter too, so no system Python is required.
- A Rust toolchain via [rustup] (`cargo`, `clippy`, `rustfmt`).

## One-time setup

```sh
uv sync
```

This creates `.venv` with the interpreter pinned in `.python-version`, builds
the extension through maturin, and installs the `dev` dependency group. maturin
builds wheels in **release** mode, so the module the Python suite imports is
optimised — benchmarks taken through `uv run` are meaningful.

## The daily loop

```sh
uv run pytest              # Python suite (rebuilds the extension if needed)
uv run ruff format         # Python formatting
uv run ruff check          # Python linting
cargo test                 # Rust unit tests and doctests
cargo clippy --all-targets -- -D warnings
cargo fmt
```

`pytest` is configured (in `pyproject.toml`) to fail if coverage of `xinterp/`
drops below 100%; `--cov-report=term-missing` prints exactly which lines are
uncovered when it does. Genuinely unreachable branches (an invariant enforced
by an earlier check) should be restructured to remove the branch rather than
carved out with a coverage pragma.

Two conveniences make this work without ceremony:

- **`uv run` rebuilds after Rust edits.** `[tool.uv] cache-keys` in
  `pyproject.toml` lists `src/**/*.rs`, `Cargo.toml` and `Cargo.lock`, so
  touching any of them makes the next `uv run` rebuild the extension before
  running. Without those keys uv would reuse a stale wheel and you would test
  the previous build.
- **`cargo` finds the interpreter on its own.** pyo3's build script needs a
  Python at least as new as its minimum supported version, and the system
  `python3` on some hosts is far older (RHEL 8 ships 3.6). `.cargo/config.toml`
  sets `PYO3_PYTHON` to `.venv/bin/python`, so `cargo test`, `cargo clippy` and
  `cargo miri` all work after `uv sync` with no environment prefix. An
  explicitly exported `PYO3_PYTHON` still wins.

If a cargo command fails inside `pyo3-build-config`, `.venv` is missing or
stale — run `uv sync` again.

## Testing against another Python

```sh
uv run --python 3.11 pytest
```

Development happens on 3.14, pinned in `.python-version`. The package supports
`>=3.11`, and CI runs the suite on the oldest and the newest supported versions.

## Lock files

`uv.lock`, `Cargo.lock` and `.python-version` are **tracked**. They make
`uv sync` reproducible and pin what the published wheels are built from. Update
them deliberately:

```sh
uv lock --upgrade          # Python dependencies
cargo update               # Rust dependencies
```

## What CI runs

`.github/workflows/CI.yml` has two gating jobs, both of which the `release` job
depends on, so a red test blocks publication to PyPI:

| job | runs |
| --- | --- |
| `rust` | `uv sync --frozen`, then `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` |
| `python` | `uv run --frozen ruff format --check`, `uv run --frozen ruff check`, `uv run --frozen pytest -q` (100% coverage required) on Python 3.11 and 3.14 |

Both use `--frozen`, so a dependency change that is not reflected in `uv.lock`
fails CI rather than silently resolving something else. Run `uv lock` and
commit the result when you change dependencies.

The remaining jobs build wheels for every supported platform and publish them
on a tag.

## Cutting a release

The version lives in **one** place: `version` in `Cargo.toml`. `pyproject.toml`
declares `dynamic = ["version"]`, so maturin reads it from the crate, and
`xinterp.__version__` reads it back from the installed distribution metadata.
There is nothing else to bump -- `uv.lock` records the root as an editable
source with no version, and `Cargo.lock` picks the new one up on the next cargo
command.

1. Bump `version` in `Cargo.toml`, then run `cargo check` so `Cargo.lock`
   follows. Commit both.
2. In `CHANGELOG.md`, give the release its own dated heading -- move anything
   still sitting under `Unreleased` into it, stamp the date as `## [0.2.0] -
   YYYY-MM-DD`, and update the compare links at the bottom.
3. Run both suites locally: `cargo test`, and `uv run pytest`.
4. Merge to `main`, then tag that commit and push the tag:

```sh
git tag 0.2.0
git push origin 0.2.0
```

The tag is what publishes. CI reruns the two gating jobs, builds wheels for
every supported platform, attests them, and uploads to PyPI only if the gates
pass -- so a red suite blocks the release rather than shipping a broken wheel.

## Conventions

- Keep `cargo clippy --all-targets -- -D warnings` clean; CI gates on it.
- Keep `ruff format`/`ruff check` clean and Python coverage at 100%; CI gates
  on both.
- Wide arithmetic is intentional: the integer path maps signed values onto
  unsigned ones (`schemes.rs`, `ToUnsigned`) and computes in `u128`, so that no
  `i64` value at any `u64` index can overflow. New kernels should follow that
  convention rather than introduce a second one.
- pyo3 module functions use the declarative form (`#[pymodule] mod rust` with
  `#[pyfunction]`); the older `#[pyfn(m)]` attribute is deprecated upstream.

[uv]: https://docs.astral.sh/uv/
[maturin]: https://www.maturin.rs/
[rustup]: https://rustup.rs/
