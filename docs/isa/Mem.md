# Memory (GM <-> Tile)

This page groups the global-memory load/store instructions.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TLOAD` | Load a tile from global memory | `docs/isa/TLOAD.md` |
| `TSTORE` | Store a tile to global memory (optionally atomic / vector / etc.) | `docs/isa/TSTORE.md` |
| `TSTORE_FP` | Store an accumulator tile with fp scaling | `docs/isa/TSTORE_FP.md` |
