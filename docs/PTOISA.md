# PTO ISA Overview

This page is a high-level overview of the PTO Tile Lib instruction set and acts as a "contents" page for the `docs/` tree.

The per-instruction reference pages in `docs/isa/` are written against the C++ intrinsic API exposed by `pto::T*` functions and use a shared structure and notation.

## Docs Contents

| Area | Page | Description |
|---|---|---|
| Overview | [`docs/README.md`](README.md) | PTO ISA guide entry point and navigation. |
| Overview | [`docs/PTOISA.md`](PTOISA.md) | This page (overview + full instruction index). |
| Getting started | [`docs/getting-started.md`](getting-started.md) | Build and run basics (recommended: start on CPU). |
| ISA reference | [`docs/isa/README.md`](isa/README.md) | Per-instruction reference directory index. |
| ISA reference | [`docs/isa/conventions.md`](isa/conventions.md) | Shared notation, operands, events, and modifiers. |
| Assembly (PTO-AS) | [`docs/grammar/README.md`](grammar/README.md) | PTO-AS documentation directory index. |
| Assembly (PTO-AS) | [`docs/grammar/PTO-AS.md`](grammar/PTO-AS.md) | PTO-AS syntax reference. |
| Assembly (PTO-AS) | [`docs/grammar/conventions.md`](grammar/conventions.md) | Grammar notation and conventions. |
| Developer notes | [`docs/coding/README.md`](coding/README.md) | Extension and implementation notes. |
| Developer notes | [`docs/coding/Tile.md`](coding/Tile.md) | Tile types, layouts, and usage. |
| Developer notes | [`docs/coding/GlobalTensor.md`](coding/GlobalTensor.md) | GlobalTensor (GM) wrapper and layouts. |
| Developer notes | [`docs/coding/Scalar.md`](coding/Scalar.md) | Scalar wrapper and immediates. |
| Developer notes | [`docs/coding/Event.md`](coding/Event.md) | Event tokens and synchronization. |
| Developer notes | [`docs/coding/ProgrammingModel.md`](coding/ProgrammingModel.md) | PTO-Auto vs PTO-Manual overview. |
| Machine model | [`docs/machine/abstract-machine.md`](machine/abstract-machine.md) | Abstract execution machine model. |
| Source of truth | [`include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp) | C++ intrinsic API (authoritative). |

## Programming Model

The ISA reference assumes the following programmer-visible models:

- Tile and valid-region semantics: `docs/coding/Tile.md`
- Global memory tensor model: `docs/coding/GlobalTensor.md`
- Event and synchronization model: `docs/coding/Event.md`
- Scalar values, type mnemonics, and enums: `docs/coding/Scalar.md`
- PTO-Auto vs PTO-Manual overview: `docs/coding/ProgrammingModel.md`

For the abstract execution model (how tile blocks are scheduled across cores/devices), see `docs/machine/abstract-machine.md`.

## Instruction Categories

| Category | Description |
|---|---|
| Synchronization | Execution-ordering primitives (event waits and pipeline barriers). |
| Manual / Resource Binding | Manual-mode primitives for binding tiles/resources to implementation-defined addresses. |
| Elementwise (Tile-Tile) | Elementwise ops that consume two tiles (tile-tile) and write one tile result. |
| Tile-Scalar / Tile-Immediate | Elementwise ops that mix a tile with a scalar/immediate value, plus a few fused patterns. |
| Axis Reduce / Expand | Reductions across a single axis, plus broadcast/expand operations that replicate per-row or per-column values. |
| Memory (GM <-> Tile) | Transfers between global memory (GM) tensors and tiles, including indirect gather/scatter. |
| Matrix Multiply | GEMM-style matrix multiply variants producing accumulator/output tiles, with optional bias/accumulation/scaling. |
| Data Movement / Layout | Tile data movement and layout transforms such as moves, transpose, reshape, and sub-tile extraction. |
| Complex | Higher-level or irregular ops such as sequence generation, index-based gather/scatter, sorting, and partial ops. |

## Instruction Index (All PTO Instructions)

This table covers all PTO instructions exposed by `include/pto/common/pto_instr.hpp` and links to the per-instruction reference pages.

| Category | Instruction | Description |
|---|---|---|
| Synchronization | [`TSYNC`](isa/TSYNC.md) | Synchronize PTO execution (wait on events or insert a per-op pipeline barrier). |
| Manual / Resource Binding | [`TASSIGN`](isa/TASSIGN.md) | Bind a Tile object to an implementation-defined on-chip address (manual placement). |
| Elementwise (Tile-Tile) | [`TABS`](isa/TABS.md) | Elementwise absolute value of a tile. |
| Elementwise (Tile-Tile) | [`TADD`](isa/TADD.md) | Elementwise add of two tiles. |
| Elementwise (Tile-Tile) | [`TADDC`](isa/TADDC.md) | Elementwise ternary add: `src0 + src1 + src2`. |
| Elementwise (Tile-Tile) | [`TAND`](isa/TAND.md) | Elementwise bitwise AND of two tiles. |
| Elementwise (Tile-Tile) | [`TCMP`](isa/TCMP.md) | Compare two tiles and write a packed predicate mask. |
| Elementwise (Tile-Tile) | [`TCVT`](isa/TCVT.md) | Elementwise type conversion with a specified rounding mode. |
| Elementwise (Tile-Tile) | [`TDIV`](isa/TDIV.md) | Elementwise division of two tiles. |
| Elementwise (Tile-Tile) | [`TEXP`](isa/TEXP.md) | Elementwise exponential. |
| Elementwise (Tile-Tile) | [`TLOG`](isa/TLOG.md) | Elementwise natural logarithm of a tile. |
| Elementwise (Tile-Tile) | [`TMAX`](isa/TMAX.md) | Elementwise maximum of two tiles. |
| Elementwise (Tile-Tile) | [`TMIN`](isa/TMIN.md) | Elementwise minimum of two tiles. |
| Elementwise (Tile-Tile) | [`TMUL`](isa/TMUL.md) | Elementwise multiply of two tiles. |
| Elementwise (Tile-Tile) | [`TNEG`](isa/TNEG.md) | Elementwise negation of a tile. |
| Elementwise (Tile-Tile) | [`TNOT`](isa/TNOT.md) | Elementwise bitwise NOT of a tile. |
| Elementwise (Tile-Tile) | [`TOR`](isa/TOR.md) | Elementwise bitwise OR of two tiles. |
| Elementwise (Tile-Tile) | [`TPRELU`](isa/TPRELU.md) | Elementwise PReLU (parametric ReLU) with a per-element slope tile. |
| Elementwise (Tile-Tile) | [`TRECIP`](isa/TRECIP.md) | Elementwise reciprocal of a tile. |
| Elementwise (Tile-Tile) | [`TRELU`](isa/TRELU.md) | Elementwise ReLU of a tile. |
| Elementwise (Tile-Tile) | [`TREM`](isa/TREM.md) | Elementwise remainder of two tiles. |
| Elementwise (Tile-Tile) | [`TRSQRT`](isa/TRSQRT.md) | Elementwise reciprocal square root. |
| Elementwise (Tile-Tile) | [`TSEL`](isa/TSEL.md) | Select between two tiles using a mask tile (per-element selection). |
| Elementwise (Tile-Tile) | [`TSHL`](isa/TSHL.md) | Elementwise shift-left of two tiles. |
| Elementwise (Tile-Tile) | [`TSHR`](isa/TSHR.md) | Elementwise shift-right of two tiles. |
| Elementwise (Tile-Tile) | [`TSQRT`](isa/TSQRT.md) | Elementwise square root. |
| Elementwise (Tile-Tile) | [`TSUB`](isa/TSUB.md) | Elementwise subtract of two tiles. |
| Elementwise (Tile-Tile) | [`TSUBC`](isa/TSUBC.md) | Elementwise ternary op: `src0 - src1 + src2`. |
| Elementwise (Tile-Tile) | [`TXOR`](isa/TXOR.md) | Elementwise bitwise XOR of two tiles. |
| Tile-Scalar / Tile-Immediate | [`TADDS`](isa/TADDS.md) | Elementwise add a scalar to a tile. |
| Tile-Scalar / Tile-Immediate | [`TADDSC`](isa/TADDSC.md) | Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`. |
| Tile-Scalar / Tile-Immediate | [`TANDS`](isa/TANDS.md) | Elementwise bitwise AND of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TCMPS`](isa/TCMPS.md) | Compare a tile against a scalar and write per-element comparison results. |
| Tile-Scalar / Tile-Immediate | [`TDIVS`](isa/TDIVS.md) | Elementwise division with a scalar (tile/scalar or scalar/tile). |
| Tile-Scalar / Tile-Immediate | [`TEXPANDS`](isa/TEXPANDS.md) | Broadcast a scalar into a destination tile. |
| Tile-Scalar / Tile-Immediate | [`TLRELU`](isa/TLRELU.md) | Leaky ReLU with a scalar slope. |
| Tile-Scalar / Tile-Immediate | [`TMAXS`](isa/TMAXS.md) | Elementwise max of a tile and a scalar: `max(src, scalar)`. |
| Tile-Scalar / Tile-Immediate | [`TMINS`](isa/TMINS.md) | Elementwise minimum of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TMULS`](isa/TMULS.md) | Elementwise multiply a tile by a scalar. |
| Tile-Scalar / Tile-Immediate | [`TORS`](isa/TORS.md) | Elementwise bitwise OR of a tile and a scalar. |
| Tile-Scalar / Tile-Immediate | [`TREMS`](isa/TREMS.md) | Elementwise remainder with a scalar: `fmod(src, scalar)` (or `%` for integers). |
| Tile-Scalar / Tile-Immediate | [`TSELS`](isa/TSELS.md) | Select one of two source tiles using a scalar `selectMode` (global select). |
| Tile-Scalar / Tile-Immediate | [`TSUBS`](isa/TSUBS.md) | Elementwise subtract a scalar from a tile. |
| Tile-Scalar / Tile-Immediate | [`TSUBSC`](isa/TSUBSC.md) | Elementwise fused op: `src0 - scalar + src1`. |
| Tile-Scalar / Tile-Immediate | [`TXORS`](isa/TXORS.md) | Elementwise bitwise XOR of a tile and a scalar. |
| Axis Reduce / Expand | [`TCOLEXPAND`](isa/TCOLEXPAND.md) | Broadcast the first element of each source column across the destination column. |
| Axis Reduce / Expand | [`TCOLMAX`](isa/TCOLMAX.md) | Reduce each column by taking the maximum across rows. |
| Axis Reduce / Expand | [`TCOLMIN`](isa/TCOLMIN.md) | Reduce each column by taking the minimum across rows. |
| Axis Reduce / Expand | [`TCOLSUM`](isa/TCOLSUM.md) | Reduce each column by summing across rows. |
| Axis Reduce / Expand | [`TROWEXPAND`](isa/TROWEXPAND.md) | Broadcast the first element of each source row across the destination row. |
| Axis Reduce / Expand | [`TROWEXPANDDIV`](isa/TROWEXPANDDIV.md) | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDMUL`](isa/TROWEXPANDMUL.md) | Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`. |
| Axis Reduce / Expand | [`TROWEXPANDSUB`](isa/TROWEXPANDSUB.md) | Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`. |
| Axis Reduce / Expand | [`TROWMAX`](isa/TROWMAX.md) | Reduce each row by taking the maximum across columns. |
| Axis Reduce / Expand | [`TROWMIN`](isa/TROWMIN.md) | Reduce each row by taking the minimum across columns. |
| Axis Reduce / Expand | [`TROWSUM`](isa/TROWSUM.md) | Reduce each row by summing across columns. |
| Memory (GM <-> Tile) | [`MGATHER`](isa/MGATHER.md) | Gather-load elements from global memory into a tile using per-element indices. |
| Memory (GM <-> Tile) | [`MSCATTER`](isa/MSCATTER.md) | Scatter-store elements from a tile into global memory using per-element indices. |
| Memory (GM <-> Tile) | [`TLOAD`](isa/TLOAD.md) | Load data from a GlobalTensor (GM) into a Tile. |
| Memory (GM <-> Tile) | [`TSTORE`](isa/TSTORE.md) | Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters. |
| Memory (GM <-> Tile) | [`TSTORE_FP`](isa/TSTORE_FP.md) | Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters. |
| Matrix Multiply | [`TMATMUL`](isa/TMATMUL.md) | Matrix multiply (GEMM) producing an accumulator/output tile. |
| Matrix Multiply | [`TMATMUL_ACC`](isa/TMATMUL_ACC.md) | Matrix multiply with accumulator input (fused accumulate). |
| Matrix Multiply | [`TMATMUL_BIAS`](isa/TMATMUL_BIAS.md) | Matrix multiply with bias add. |
| Matrix Multiply | [`TMATMUL_MX`](isa/TMATMUL_MX.md) | Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets. |
| Data Movement / Layout | [`TEXTRACT`](isa/TEXTRACT.md) | Extract a sub-tile from a source tile. |
| Data Movement / Layout | [`TMOV`](isa/TMOV.md) | Move/copy between tiles, optionally applying implementation-defined conversion modes. |
| Data Movement / Layout | [`TMOV_FP`](isa/TMOV_FP.md) | Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters. |
| Data Movement / Layout | [`TRESHAPE`](isa/TRESHAPE.md) | Reinterpret a tile as another tile type/shape while preserving the underlying bytes. |
| Data Movement / Layout | [`TTRANS`](isa/TTRANS.md) | Transpose with an implementation-defined temporary tile. |
| Complex | [`TCI`](isa/TCI.md) | Generate a contiguous integer sequence into a destination tile. |
| Complex | [`TGATHER`](isa/TGATHER.md) | Gather/select elements using either an index tile or a compile-time mask pattern. |
| Complex | [`TGATHERB`](isa/TGATHERB.md) | Gather elements using byte offsets. |
| Complex | [`TMRGSORT`](isa/TMRGSORT.md) | Merge sort for multiple sorted lists (implementation-defined element format and layout). |
| Complex | [`TPARTADD`](isa/TPARTADD.md) | Partial elementwise add with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMAX`](isa/TPARTMAX.md) | Partial elementwise max with implementation-defined handling of mismatched valid regions. |
| Complex | [`TPARTMIN`](isa/TPARTMIN.md) | Partial elementwise min with implementation-defined handling of mismatched valid regions. |
| Complex | [`TSCATTER`](isa/TSCATTER.md) | Scatter rows of a source tile into a destination tile using per-element row indices. |
| Complex | [`TSORT32`](isa/TSORT32.md) | Sort a fixed-size 32-element block and produce an index mapping. |
