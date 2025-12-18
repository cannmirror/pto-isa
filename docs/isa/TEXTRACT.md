# TEXTRACT

## Introduction

Extract a sub-tile from a source tile.

## Math Interpretation

Conceptually copies a window starting at `(indexRow, indexCol)` from `src` into `dst`. Exact mapping depends on layouts.

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## IR Syntax

Synchronous form (from `docs/ir/PTO-IR.md`):

```mlir
%dst = pto.tile.extract %src[%r0, %r1]
    : tile<SrcShape x Ts, #pto.tile_info<loc=Mat, layout=Ls>>
   -> tile<DstShape x Ts, #pto.tile_info<loc=Left|Right, layout=Ld>>
```

Asynchronous form:

```mlir
%dst, %e = pto.tile.extract %src[%r0, %r1] wait(%e0)
    : tile<...> -> tile<...>, !pto.event<producer = #pto.op<TEXTRACT>>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                              WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `DstTileData::DType` must equal `SrcTileData::DType` and must be one of: `int8_t`, `half`, `bfloat16_t`, `float`.
  - Source fractal must satisfy: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`.
  - Runtime bounds checks:
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`
  - Destination must be `TileType::Left` or `TileType::Right` with a target-supported fractal configuration.
- **Implementation checks (A5)**:
  - `DstTileData::DType` must equal `SrcTileData::DType` and must be one of: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3fn_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
  - Source fractal must satisfy: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`.
  - Destination supports `Mat -> Left/Right` and also supports `Vec -> Mat` for specific tile locations (no explicit runtime bounds assertions are enforced in `TEXTRACT_IMPL` on this target).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```
