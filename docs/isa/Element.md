# Elementwise (Tile-Tile)

This page groups tile-to-tile elementwise arithmetic, selection, and dtype conversion.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TADD` | Elementwise add | `docs/isa/TADD.md` |
| `TABS` | Elementwise absolute value | `docs/isa/TABS.md` |
| `TSUB` | Elementwise subtract | `docs/isa/TSUB.md` |
| `TMUL` | Elementwise multiply | `docs/isa/TMUL.md` |
| `TDIV` | Elementwise divide | `docs/isa/TDIV.md` |
| `TMIN` | Elementwise min | `docs/isa/TMIN.md` |
| `TMAX` | Elementwise max | `docs/isa/TMAX.md` |
| `TEXP` | Elementwise exponential | `docs/isa/TEXP.md` |
| `TSQRT` | Elementwise square root | `docs/isa/TSQRT.md` |
| `TRSQRT` | Elementwise reciprocal square root | `docs/isa/TRSQRT.md` |
| `TSEL` | Elementwise select (mask) | `docs/isa/TSEL.md` |
| `TCMP` | Tile-to-tile compare (mask) | `docs/isa/TCMP.md` |
| `TCVT` | Elementwise dtype conversion | `docs/isa/TCVT.md` |

For ops that mix tiles with scalars/immediates (for example `TADDS`, `TMULS`, `TCMPS`), see `docs/isa/TileScalar.md`.
