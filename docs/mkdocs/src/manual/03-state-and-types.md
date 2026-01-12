# 3. State and types

## 3.1 Architectural state (conceptual)

PTO programs can be described using the following conceptual state:

- **Global memory (GM)**: persistent memory visible to the device program
- **Tile storage**: on-chip 2D storage used by tile operands (locations such as `Vec`, `Mat`, `Left`, `Right`, `Acc`, …)
- **Scalar state**: scalar registers/immediates used as modifiers, strides, modes, etc.
- **Synchronization state**: events/flags used to order pipelines and cross-core handshakes

The exact physical mapping is platform-dependent; the ISA defines the observable behavior.

## 3.2 Element types

Instructions are parameterized by element types (e.g. `fp16`, `bf16`, `fp32`, `s32`, …). Type legality is instruction-specific.

For the canonical type tables and naming, see:

- `docs/PTOISA.md`
- `docs/isa/conventions.md`

## 3.3 Tile operands

A tile operand conceptually has:

- **Location**: selects the storage class / functional unit intent (e.g. `Vec` vs `Left`)
- **Element type**
- **Shape**: `[R, C]` (usually compile-time constants)
- **Layout metadata**
- **Valid region**: `[Rv, Cv]` (mask)

See: `docs/coding/Tile.md`

## 3.4 GlobalTensor operands

A GlobalTensor is a typed view over GM with:

- element type
- shape (up to 5D in the library model)
- stride
- an optional layout hint

See: `docs/coding/GlobalTensor.md`

