<!--
Guidance for AI coding agents working on the `bayesfm` repository.
Keep this file concise (20-50 lines). It focuses on the project's structure, build/test/debug workflows,
and codebase-specific conventions that matter when editing or generating code.
-->

# AI Assistant Instructions — bayesfm (Rust)

This project is a Rust crate for Bayesian forward modeling with optional Python bindings via `pyo3`.
When making changes, prefer small, targeted edits and preserve public APIs unless a refactor is explicit.

Key points to be immediately productive:

- Big picture: core crates and intent
  - `src/lib.rs` is the public crate root. Major modules: `conf`, `geometry`, `methods`, `math`, `obs`, `noise`, and optional `pytypes` (enabled by the `pyo3` feature).
  - Models implement the `Model<T, D, P>` trait (in `src/lib.rs`) and often expose ensemble helpers (`ensbl.rs`). Many algorithms operate generically over numeric field type `T` and const generics for dimensions/parameters.
  - `methods/` contains algorithmic routines (e.g. `fisher.rs`), while `methods/filters` implements filtering/ABC/SIR-style algorithms.

- Build / test / run workflows
  - Build: `cargo build` (uses Rust edition 2024, rust >= 1.92). Release: `cargo build --release`.
  - Run examples/tests: `cargo test` runs unit tests. Benchmarks use `criterion` in Cargo dependencies — run with `cargo bench`.
  - To enable Python bindings: build with feature `pyo3` (and system Python dev headers): `cargo build --features pyo3` (or `--features pyo3 --release`). The `numpy` and `paste` crates are optional and tied to the `pyo3` feature.

- Important conventions and patterns
  - Determinism: `Model::evolve_fmst` must be deterministic and must not use RNG. RNG belongs in `initialize_*` methods or sampling utilities.
  - Const generics: many types and functions use const generics for dimensions (`D`) and parameter counts (`P`). Maintain type signatures when editing generic code.
  - Observation time-series: `ConfSeries<OC>` (in `src/conf`) represents single or combined observers; `composite_indices` tracks source observers. Use `combine()`/`uncombine()` where appropriate.
  - Ensemble APIs: `EnsembleState`, `EnsembleObservations`, and related helpers are used across `methods` to compute finite-difference ensembles (see `methods/fisher.rs`). Follow their runtime patterns when adding ensemble algorithms.

- Cross-language integration
  - Python bindings live under `src/pytypes/` and are gated behind the `pyo3` feature. When modifying pyo3-exposed types, keep Python-friendly naming and Float aliasing in `pytypes/mod.rs` (f32 via `pyo3_f32` feature, otherwise f64).

- Typical edit checklist for AI changes
  - Update public doc comments in the item you change (the crate has `missing_docs = "deny"` lint enabled).
  - Run `cargo build` and `cargo test` locally — fix type/lint errors before proposing PRs.
  - If you change API shapes (traits, public structs), add or update a minimal unit test demonstrating the new behavior.

- Files to reference when reasoning about design or adding features
  - `src/lib.rs` — core trait definitions and crate exports
  - `src/conf/` — `ConfSeries` and observer semantics
  - `src/methods/fisher.rs` — example of ensemble finite-difference patterns and use of `prodef::MultivariateNormalDensity`
  - `src/pytypes/` — Python exposure patterns and Float aliasing
  - `Cargo.toml` — features and optional deps (pyo3, numpy, paste)

If anything in these instructions is unclear, tell me which part (architecture, build, or conventions) you want expanded and I will iterate.
