# PTO ISA Conventions

This page defines notation shared by all instruction docs in `docs/isa/`.

## Where the ISA is defined

The public C++ intrinsics are declared in `include/pto/common/pto_instr.hpp`. The docs in this folder describe those APIs.

## Operands and notation

- **Tile**: A small tensor stored in on-chip tile storage. Tiles are declared via `pto::Tile<...>` in `include/pto/common/pto_tile.hpp`.
- **GlobalTensor (GM)**: A tensor in global memory. `TLOAD`/`TSTORE` move data between GM and Tiles.
- **`dst`, `src0`, `src1`**: Destination / source tile operands.
- **Scalar operand**: A scalar value passed as a C++ argument (often `typename Tile::DType`, but some ops accept any `T` convertible to the tile element type).

## Valid region and padding

Most vector-style operations conceptually operate on the **valid region** of a tile (runtime `tile.GetValidRow()` / `tile.GetValidCol()`), and may ignore elements outside it. Padding behavior for out-of-valid elements is tile-type / implementation dependent.

## Events and synchronization

Most intrinsics in `include/pto/common/pto_instr.hpp` have a trailing `WaitEvents&... events` parameter pack and internally call `TSYNC(events...)` before issuing the instruction.

- `pto::RecordEvent` is the return type used by many intrinsics to support event recording via assignment.
- `pto::Event<SrcOp, DstOp>` (defined in `include/pto/common/event.hpp`) can record/wait between pipeline operations.

Example:

```cpp
pto::Event<pto::Op::TLOAD, pto::Op::TADD> ev;
ev = pto::TLOAD(tile0, global0);
pto::TADD(dst, tile0, tile1, ev);
```

### `TSYNC`

- `TSYNC<pto::Op::...>()` inserts a pipeline barrier for a single op (vector ops only).
- `TSYNC(ev0, ev1, ...)` waits for all provided events.

### `TSORT32` special case

`TSORT32(...)` does **not** take `WaitEvents&...` and does **not** call `TSYNC(...)` internally. If you need synchronization, call `TSYNC(...)` explicitly before `TSORT32`.

## Common modifiers and enums

- `pto::CmpMode` (in `include/pto/common/type.hpp`): `EQ`, `NE`, `LT`, `GT`, `GE`, `LE`.
- `pto::RoundMode` (in `include/pto/common/constants.hpp`): `CAST_RINT`, `CAST_ROUND`, `CAST_FLOOR`, `CAST_CEIL`, `CAST_TRUNC`, `CAST_ODD` (and `CAST_NONE`).
- `pto::MaskPattern` (in `include/pto/common/type.hpp`): pattern constants used by some `TGATHER` variants.
- `pto::AccToVecMode`, `pto::ReluPreMode`, `pto::AtomicType` (in `include/pto/common/constants.hpp`) are used by `TMOV` and `TSTORE`.

