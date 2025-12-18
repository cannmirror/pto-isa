# Tile-Scalar / Tile-Immediate

This page groups instructions that combine a tile with a scalar/immediate parameter.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TADDS` | Tile + scalar | `docs/isa/TADDS.md` |
| `TDIVS` | Tile / scalar (and scalar / tile variants depending on target) | `docs/isa/TDIVS.md` |
| `TMULS` | Tile * scalar | `docs/isa/TMULS.md` |
| `TMINS` | Elementwise min(tile, scalar) | `docs/isa/TMINS.md` |
| `TCMPS` | Compare(tile, scalar) producing a mask tile | `docs/isa/TCMPS.md` |
| `TEXPANDS` | Expand a scalar into a tile | `docs/isa/TEXPANDS.md` |
| `TSELS` | Select using an immediate/mode | `docs/isa/TSELS.md` |
