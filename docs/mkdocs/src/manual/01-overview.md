# 1. Overview

## 1.1 What PTO programs operate on

At ISA level, PTO is a **tile-oriented** ISA:

- A **Tile** is a 2D on-chip operand (conceptually a small matrix with metadata).
- A **GlobalTensor** is a view of data in global memory (GM), also treated as a typed, shaped object for load/store.

The ISA is built around moving data between these objects and applying compute over 2D tile regions.

See also:

- Tile model: `docs/coding/Tile.md`
- GlobalTensor model: `docs/coding/GlobalTensor.md`

## 1.2 Instruction families

PTO instructions are grouped by function:

- **Data movement**: `TLOAD`, `TSTORE`, `TASSIGN`, `TEXTRACT`, `TMOV`, `TTRANS`, `TRESHAPE`
- **Elementwise / scalar-tile**: `TADD`, `TSUB`, `TMUL`, `TDIV`, `TEXP`, `TLOG`, etc.
- **Reductions / expands**: `TROWMAX`, `TROWSUM`, `TROWEXPAND`, `TCOLSUM`, etc.
- **Matrix / cube**: `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, …
- **Predication / selection**: `TCMP`, `TCMPS`, `TSEL`, `TSELS`
- **Gather/scatter**: `TGATHER`, `TSCATTER`, `MGATHER`, `MSCATTER`
- **Synchronization**: `TSYNC` and event/flag primitives used by backends

The authoritative index is `docs/isa/README.md`.

## 1.3 Valid region and masks (the core semantic rule)

Many PTO operands include a **valid region** (mask) describing which elements are semantically meaningful. This manual uses:

- `shape(T) = [R, C]` for the physical tile size
- `valid(T) = [Rv, Cv]` for the valid region size (typically `Rv ≤ R`, `Cv ≤ C`)

Unless an instruction defines special behavior:

- elementwise operations are defined only for indices `(r, c)` within the valid region
- behavior outside the valid region is backend-dependent (may preserve, may zero, may be undefined)

If you need a deterministic behavior for out-of-range elements, explicitly materialize it (e.g. `TFILLPAD`) or use a definition that specifies padding.

## 1.4 Where this manual fits

- Architectural narrative and shared rules: **this manual**
- Per-instruction semantics, operand constraints, examples: `docs/isa/*.md`
- Assembly syntax / grammar: `docs/grammar/PTO-AS.md`
- Execution/machine abstraction: `docs/machine/abstract-machine.md`

