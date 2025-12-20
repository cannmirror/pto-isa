# PTO ISA Overview

This page is a high-level overview of the PTO Tile Lib instruction set. For the authoritative reference, use:

- Instruction index: [`docs/isa/README.md`](isa/README.md)
- Shared conventions: [`docs/isa/conventions.md`](isa/conventions.md)
- PTO assembly syntax reference (PTO-AS): [`docs/grammar/PTO-AS.md`](grammar/PTO-AS.md)
- Source of truth (C++ intrinsics): [`include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp)

The per-instruction reference pages in `docs/isa/` are written against the C++ intrinsic API exposed by `pto::T*` functions and use a shared structure and notation.

## Instruction Index (All PTO Instructions)

This list covers all PTO instructions exposed by `include/pto/common/pto_instr.hpp` and links to the per-instruction reference pages.

### Synchronization

- [`TSYNC`](isa/TSYNC.md)

### Manual / Resource Binding

- [`TASSIGN`](isa/TASSIGN.md)

### Elementwise (Tile-Tile)

- [`TADD`](isa/TADD.md)
- [`TABS`](isa/TABS.md)
- [`TSUB`](isa/TSUB.md)
- [`TMUL`](isa/TMUL.md)
- [`TDIV`](isa/TDIV.md)
- [`TREM`](isa/TREM.md)
- [`TSHL`](isa/TSHL.md)
- [`TSHR`](isa/TSHR.md)
- [`TAND`](isa/TAND.md)
- [`TOR`](isa/TOR.md)
- [`TXOR`](isa/TXOR.md)
- [`TMIN`](isa/TMIN.md)
- [`TMAX`](isa/TMAX.md)
- [`TEXP`](isa/TEXP.md)
- [`TLOG`](isa/TLOG.md)
- [`TSQRT`](isa/TSQRT.md)
- [`TRSQRT`](isa/TRSQRT.md)
- [`TRECIP`](isa/TRECIP.md)
- [`TNEG`](isa/TNEG.md)
- [`TNOT`](isa/TNOT.md)
- [`TRELU`](isa/TRELU.md)
- [`TPRELU`](isa/TPRELU.md)
- [`TADDC`](isa/TADDC.md)
- [`TSUBC`](isa/TSUBC.md)
- [`TSEL`](isa/TSEL.md)
- [`TCMP`](isa/TCMP.md)
- [`TCVT`](isa/TCVT.md)

### Tile-Scalar / Tile-Immediate

- [`TADDS`](isa/TADDS.md)
- [`TSUBS`](isa/TSUBS.md)
- [`TDIVS`](isa/TDIVS.md)
- [`TMULS`](isa/TMULS.md)
- [`TREMS`](isa/TREMS.md)
- [`TMAXS`](isa/TMAXS.md)
- [`TMINS`](isa/TMINS.md)
- [`TANDS`](isa/TANDS.md)
- [`TORS`](isa/TORS.md)
- [`TXORS`](isa/TXORS.md)
- [`TCMPS`](isa/TCMPS.md)
- [`TEXPANDS`](isa/TEXPANDS.md)
- [`TSELS`](isa/TSELS.md)
- [`TLRELU`](isa/TLRELU.md)
- [`TADDSC`](isa/TADDSC.md)
- [`TSUBSC`](isa/TSUBSC.md)

### Axis Reduce / Expand

- [`TROWSUM`](isa/TROWSUM.md)
- [`TROWMAX`](isa/TROWMAX.md)
- [`TROWMIN`](isa/TROWMIN.md)
- [`TROWEXPAND`](isa/TROWEXPAND.md)
- [`TROWEXPANDDIV`](isa/TROWEXPANDDIV.md)
- [`TROWEXPANDMUL`](isa/TROWEXPANDMUL.md)
- [`TROWEXPANDSUB`](isa/TROWEXPANDSUB.md)
- [`TCOLSUM`](isa/TCOLSUM.md)
- [`TCOLMAX`](isa/TCOLMAX.md)
- [`TCOLMIN`](isa/TCOLMIN.md)
- [`TCOLEXPAND`](isa/TCOLEXPAND.md)

### Memory (GM <-> Tile)

- [`TLOAD`](isa/TLOAD.md)
- [`TSTORE`](isa/TSTORE.md)
- [`TSTORE_FP`](isa/TSTORE_FP.md)
- [`MGATHER`](isa/MGATHER.md)
- [`MSCATTER`](isa/MSCATTER.md)

### Matrix Multiply

- [`TMATMUL`](isa/TMATMUL.md)
- [`TMATMUL_MX`](isa/TMATMUL_MX.md)
- [`TMATMUL_ACC`](isa/TMATMUL_ACC.md)
- [`TMATMUL_BIAS`](isa/TMATMUL_BIAS.md)

### Data Movement / Layout

- [`TMOV`](isa/TMOV.md)
- [`TMOV_FP`](isa/TMOV_FP.md)
- [`TTRANS`](isa/TTRANS.md)
- [`TEXTRACT`](isa/TEXTRACT.md)
- [`TRESHAPE`](isa/TRESHAPE.md)

### Complex

- [`TCI`](isa/TCI.md)
- [`TGATHER`](isa/TGATHER.md)
- [`TGATHERB`](isa/TGATHERB.md)
- [`TSCATTER`](isa/TSCATTER.md)
- [`TSORT32`](isa/TSORT32.md)
- [`TMRGSORT`](isa/TMRGSORT.md)
- [`TPARTADD`](isa/TPARTADD.md)
- [`TPARTMAX`](isa/TPARTMAX.md)
- [`TPARTMIN`](isa/TPARTMIN.md)
