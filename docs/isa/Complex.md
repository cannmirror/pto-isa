# Complex Ops

This page groups higher-level tile ops such as gather, sort, and partitioned reductions.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TCI` | Generate a consecutive integer sequence tile | `docs/isa/TCI.md` |
| `TGATHER` | Gather / masked gather | `docs/isa/TGATHER.md` |
| `TGATHERB` | Gather with per-element offsets | `docs/isa/TGATHERB.md` |
| `TSCATTER` | Scatter using per-element indices | `docs/isa/TSCATTER.md` |
| `TSORT32` | Sort 32 elements (value + index) | `docs/isa/TSORT32.md` |
| `TMRGSORT` | Merge-sort across multiple tiles | `docs/isa/TMRGSORT.md` |
| `TPARTADD` | Partitioned add (segmented) | `docs/isa/TPARTADD.md` |
| `TPARTMAX` | Partitioned max (segmented) | `docs/isa/TPARTMAX.md` |
| `TPARTMIN` | Partitioned min (segmented) | `docs/isa/TPARTMIN.md` |
