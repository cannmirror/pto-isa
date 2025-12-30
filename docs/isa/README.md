<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Reference

This directory contains the per-instruction reference for the PTO Tile Lib ISA.

- Source of truth (C++ intrinsics): [`docs/reference/pto-intrinsics-header.md`](../reference/pto-intrinsics-header.md) (declared in `include/pto/common/pto_instr.hpp`).
- Common conventions (operands, events, modifiers): [`docs/isa/conventions.md`](conventions.md)


## Elementwise (Tile-Tile)
- [`TADD`](TADD.md) — Elementwise add of two tiles.
- [`TABS`](TABS.md) — Elementwise absolute value of a tile.
- [`TSUB`](TSUB.md) — Elementwise subtract of two tiles.
- [`TMUL`](TMUL.md) — Elementwise multiply of two tiles.
- [`TDIV`](TDIV.md) — Elementwise division of two tiles.
- [`TREM`](TREM.md) — Elementwise remainder of two tiles.
- [`TSHL`](TSHL.md) — Elementwise shift-left of two tiles.
- [`TSHR`](TSHR.md) — Elementwise shift-right of two tiles.
- [`TAND`](TAND.md) — Elementwise bitwise AND of two tiles.
- [`TOR`](TOR.md) — Elementwise bitwise OR of two tiles.
- [`TXOR`](TXOR.md) — Elementwise bitwise XOR of two tiles.
- [`TMIN`](TMIN.md) — Elementwise minimum of two tiles.
- [`TMAX`](TMAX.md) — Elementwise maximum of two tiles.
- [`TEXP`](TEXP.md) — Elementwise exponential.
- [`TLOG`](TLOG.md) — Elementwise natural logarithm of a tile.
- [`TSQRT`](TSQRT.md) — Elementwise square root.
- [`TRSQRT`](TRSQRT.md) — Elementwise reciprocal square root.
- [`TRECIP`](TRECIP.md) — Elementwise reciprocal of a tile.
- [`TNEG`](TNEG.md) — Elementwise negation of a tile.
- [`TNOT`](TNOT.md) — Elementwise bitwise NOT of a tile.
- [`TRELU`](TRELU.md) — Elementwise ReLU of a tile.
- [`TPRELU`](TPRELU.md) — Elementwise PReLU (parametric ReLU) with a per-element slope tile.
- [`TADDC`](TADDC.md) — Elementwise ternary add: `src0 + src1 + src2`.
- [`TSUBC`](TSUBC.md) — Elementwise ternary op: `src0 - src1 + src2`.
- [`TSEL`](TSEL.md) — Select between two tiles using a mask tile (per-element selection).
- [`TCMP`](TCMP.md) — Compare two tiles and write a packed predicate mask.
- [`TCVT`](TCVT.md) — Elementwise type conversion with a specified rounding mode.

## Tile-Scalar / Tile-Immediate
- [`TADDS`](TADDS.md) — Elementwise add a scalar to a tile.
- [`TSUBS`](TSUBS.md) — Elementwise subtract a scalar from a tile.
- [`TDIVS`](TDIVS.md) — Elementwise division with a scalar (tile/scalar or scalar/tile).
- [`TMULS`](TMULS.md) — Elementwise multiply a tile by a scalar.
- [`TREMS`](TREMS.md) — Elementwise remainder with a scalar: `fmod(src, scalar)` (or `%` for integers).
- [`TMAXS`](TMAXS.md) — Elementwise max of a tile and a scalar: `max(src, scalar)`.
- [`TMINS`](TMINS.md) — Elementwise minimum of a tile and a scalar.
- [`TANDS`](TANDS.md) — Elementwise bitwise AND of a tile and a scalar.
- [`TORS`](TORS.md) — Elementwise bitwise OR of a tile and a scalar.
- [`TXORS`](TXORS.md) — Elementwise bitwise XOR of a tile and a scalar.
- [`TCMPS`](TCMPS.md) — Compare a tile against a scalar and write per-element comparison results.
- [`TEXPANDS`](TEXPANDS.md) — Broadcast a scalar into a destination tile.
- [`TSELS`](TSELS.md) — Select one of two source tiles using a scalar `selectMode` (global select).
- [`TLRELU`](TLRELU.md) — Leaky ReLU with a scalar slope.
- [`TADDSC`](TADDSC.md) — Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`.
- [`TSUBSC`](TSUBSC.md) — Elementwise fused op: `src0 - scalar + src1`.

## Axis Reduce / Expand
- [`TROWSUM`](TROWSUM.md) — Reduce each row by summing across columns.
- [`TROWMAX`](TROWMAX.md) — Reduce each row by taking the maximum across columns.
- [`TROWMIN`](TROWMIN.md) — Reduce each row by taking the minimum across columns.
- [`TROWEXPAND`](TROWEXPAND.md) — Broadcast the first element of each source row across the destination row.
- [`TROWEXPANDDIV`](TROWEXPANDDIV.md) — Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`.
- [`TROWEXPANDMUL`](TROWEXPANDMUL.md) — Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.
- [`TROWEXPANDSUB`](TROWEXPANDSUB.md) — Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`.
- [`TCOLSUM`](TCOLSUM.md) — Reduce each column by summing across rows.
- [`TCOLMAX`](TCOLMAX.md) — Reduce each column by taking the maximum across rows.
- [`TCOLMIN`](TCOLMIN.md) — Reduce each column by taking the minimum across rows.
- [`TCOLEXPAND`](TCOLEXPAND.md) — Broadcast the first element of each source column across the destination column.

## Padding
- [`TFILLPAD`](TFILLPAD.md) — Copy a source tile into a destination tile and fill the remaining (padded) elements with a compile-time pad value selected by `TileDataDst::PadVal` (e.g., `PadValue::Min`/`PadValue::Max`).

## Memory (GM <-> Tile)
- [`TLOAD`](TLOAD.md) — Load data from a GlobalTensor (GM) into a Tile.
- [`TSTORE`](TSTORE.md) — Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters.
- [`TSTORE_FP`](TSTORE_FP.md) — Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters.
- [`MGATHER`](MGATHER.md) — Gather-load elements from global memory into a tile using per-element indices.
- [`MSCATTER`](MSCATTER.md) — Scatter-store elements from a tile into global memory using per-element indices.

## Matrix Multiply
- [`TMATMUL`](TMATMUL.md) — Matrix multiply (GEMM) producing an accumulator/output tile.
- [`TMATMUL_MX`](TMATMUL_MX.md) — Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets.
- [`TMATMUL_ACC`](TMATMUL_ACC.md) — Matrix multiply with accumulator input (fused accumulate).
- [`TMATMUL_BIAS`](TMATMUL_BIAS.md) — Matrix multiply with bias add.

## Data Movement / Layout
- [`TMOV`](TMOV.md) — Move/copy between tiles, optionally applying implementation-defined conversion modes.
- [`TMOV_FP`](TMOV_FP.md) — Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters.
- [`TTRANS`](TTRANS.md) — Transpose with an implementation-defined temporary tile.
- [`TEXTRACT`](TEXTRACT.md) — Extract a sub-tile from a source tile.
- [`TRESHAPE`](TRESHAPE.md) — Reinterpret a tile as another tile type/shape while preserving the underlying bytes.
- [`TASSIGN`](TASSIGN.md) — Bind a Tile object to an implementation-defined on-chip address (manual placement).

## Complex
- [`TCI`](TCI.md) — Generate a contiguous integer sequence into a destination tile.
- [`TGATHER`](TGATHER.md) — Gather/select elements using either an index tile or a compile-time mask pattern.
- [`TGATHERB`](TGATHERB.md) — Gather elements using byte offsets.
- [`TSCATTER`](TSCATTER.md) — Scatter rows of a source tile into a destination tile using per-element row indices.
- [`TSORT32`](TSORT32.md) — Sort a fixed-size 32-element block and produce an index mapping.
- [`TMRGSORT`](TMRGSORT.md) — Merge sort for multiple sorted lists (implementation-defined element format and layout).
- [`TPARTADD`](TPARTADD.md) — Partial elementwise add with implementation-defined handling of mismatched valid regions.
- [`TPARTMAX`](TPARTMAX.md) — Partial elementwise max with implementation-defined handling of mismatched valid regions.
- [`TPARTMIN`](TPARTMIN.md) — Partial elementwise min with implementation-defined handling of mismatched valid regions.

## Synchronization
- [`TSYNC`](TSYNC.md) — Synchronize PTO execution (wait on events or insert a per-op pipeline barrier).
