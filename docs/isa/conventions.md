# PTO ISA Conventions

This page defines shared conventions used by the per-instruction ISA reference pages in `docs/isa/` and the corresponding C++ intrinsics in `include/pto/common/pto_instr.hpp`.

## Notation

- **Tile**: A fixed-size on-chip tile object (e.g., `pto::Tile<...>`). Many instructions operate on tiles and use the tile’s valid region (`GetValidRow()`, `GetValidCol()`).
- **GM (global memory)**: Off-chip memory accessed via `pto::GlobalTensor<...>`.
- **Scalar / immediate**: A host-side scalar value or an encoded immediate used by `*S` / `*C` variants.

## Shapes and layouts

- **Row-major vs. column-major**: Unless stated otherwise, CPU simulator kernels assume row-major tiles. Instructions that support multiple layouts will state supported layouts explicitly.
- **Valid region**: The effective compute region is `(valid_row, valid_col)` as provided by the destination tile (or specified in the instruction description). Elements outside the valid region are unspecified unless the instruction says otherwise.

## Types

- The instruction page lists supported data types (e.g., `fp16`, `fp32`, `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`). CPU simulator support may be a subset and is documented in `include/README.md`.

## Events and synchronization

- Instructions may require ordering between memory and vector pipelines. When examples show events (e.g., `set_flag(...)` / `wait_flag(...)`), they indicate the required ordering constraints on the target backend.
- `TSYNC` is used for explicit synchronization when needed by a sequence of instructions.

