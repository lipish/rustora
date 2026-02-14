# rustora

A minimal Rust-first foundation for building typed AI agents inspired by Pydantic AI.

## Current MVP surface

- `Agent<Deps, Output, Model>` generic core
- Structured JSON output validation via `serde` + `schemars`
- Built-in reflection loop on validation failures
- Tool trait with automatic input JSON Schema generation
