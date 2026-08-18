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
cargo test                 # Rust unit tests and doctests
cargo clippy --all-targets -- -D warnings
cargo fmt
```

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
| `python` | `uv run --frozen pytest -q` on Python 3.11 and 3.14 |

Both use `--frozen`, so a dependency change that is not reflected in `uv.lock`
fails CI rather than silently resolving something else. Run `uv lock` and
commit the result when you change dependencies.

The remaining jobs build wheels for every supported platform and publish them
on a tag.

## Conventions

- Keep `cargo clippy --all-targets -- -D warnings` clean; CI gates on it.
- Wide arithmetic is intentional: the integer path maps signed values onto
  unsigned ones (`schemes.rs`, `ToUnsigned`) and computes in `u128`, so that no
  `i64` value at any `u64` index can overflow. New kernels should follow that
  convention rather than introduce a second one.
- pyo3 module functions use the declarative form (`#[pymodule] mod rust` with
  `#[pyfunction]`); the older `#[pyfn(m)]` attribute is deprecated upstream.

[uv]: https://docs.astral.sh/uv/
[maturin]: https://www.maturin.rs/
[rustup]: https://rustup.rs/
