# Data Movement / Layout

This page groups tile movement and layout-related instructions.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TMOV` | Move/convert between tile locations/types (Mat/Left/Right/Bias/Acc/Vec) | `docs/isa/TMOV.md` |
| `TMOV_FP` | Acc->Vec/Mat move using fp scaling | `docs/isa/TMOV_FP.md` |
| `TTRANS` | Transpose a tile | `docs/isa/TTRANS.md` |
| `TEXTRACT` | Extract a sub-tile | `docs/isa/TEXTRACT.md` |
