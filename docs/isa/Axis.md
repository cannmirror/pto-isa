# Axis Reduce / Expand

This page groups instructions that reduce over a row/column dimension, or broadcast a reduced value back across an axis.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TROWSUM` | Row-wise sum reduction | `docs/isa/TROWSUM.md` |
| `TROWMAX` | Row-wise max reduction | `docs/isa/TROWMAX.md` |
| `TROWMIN` | Row-wise min reduction | `docs/isa/TROWMIN.md` |
| `TROWEXPAND` | Broadcast each row's first element across the row | `docs/isa/TROWEXPAND.md` |
| `TROWEXPANDDIV` | Row-wise broadcast divide | `docs/isa/TROWEXPANDDIV.md` |
| `TROWEXPANDMUL` | Row-wise broadcast multiply | `docs/isa/TROWEXPANDMUL.md` |
| `TROWEXPANDSUB` | Row-wise broadcast subtract | `docs/isa/TROWEXPANDSUB.md` |
| `TCOLSUM` | Column-wise sum reduction | `docs/isa/TCOLSUM.md` |
| `TCOLMAX` | Column-wise max reduction | `docs/isa/TCOLMAX.md` |
| `TCOLMIN` | Column-wise min reduction | `docs/isa/TCOLMIN.md` |
