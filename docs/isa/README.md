# PTO ISA Reference

This directory contains the per-instruction reference for the PTO Tile Lib ISA.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Common conventions (operands, events, modifiers): `docs/isa/conventions.md`

## Category Pages (Overview)
- Elementwise: `docs/isa/Element.md`
- Tile-Scalar / Tile-Immediate: `docs/isa/TileScalar.md`
- Axis Reduce / Expand: `docs/isa/Axis.md`
- Memory (GM <-> Tile): `docs/isa/Mem.md`
- Matrix Multiply: `docs/isa/Matmul.md`
- Data Movement / Layout: `docs/isa/FixPipe.md`
- Complex: `docs/isa/Complex.md`
- Manual / Resource Binding: `docs/isa/Manual.md`

## Elementwise (Tile-Tile)
- `TADD`: `docs/isa/TADD.md`
- `TABS`: `docs/isa/TABS.md`
- `TSUB`: `docs/isa/TSUB.md`
- `TMUL`: `docs/isa/TMUL.md`
- `TDIV`: `docs/isa/TDIV.md`
- `TMIN`: `docs/isa/TMIN.md`
- `TMAX`: `docs/isa/TMAX.md`
- `TEXP`: `docs/isa/TEXP.md`
- `TSQRT`: `docs/isa/TSQRT.md`
- `TRSQRT`: `docs/isa/TRSQRT.md`
- `TSEL`: `docs/isa/TSEL.md`
- `TCMP`: `docs/isa/TCMP.md`
- `TCVT`: `docs/isa/TCVT.md`

## Tile-Scalar / Tile-Immediate
- `TADDS`: `docs/isa/TADDS.md`
- `TDIVS`: `docs/isa/TDIVS.md`
- `TMULS`: `docs/isa/TMULS.md`
- `TMINS`: `docs/isa/TMINS.md`
- `TCMPS`: `docs/isa/TCMPS.md`
- `TEXPANDS`: `docs/isa/TEXPANDS.md`
- `TSELS`: `docs/isa/TSELS.md`

## Axis Reduce / Expand
- `TROWSUM`: `docs/isa/TROWSUM.md`
- `TROWMAX`: `docs/isa/TROWMAX.md`
- `TROWMIN`: `docs/isa/TROWMIN.md`
- `TROWEXPAND`: `docs/isa/TROWEXPAND.md`
- `TROWEXPANDDIV`: `docs/isa/TROWEXPANDDIV.md`
- `TROWEXPANDMUL`: `docs/isa/TROWEXPANDMUL.md`
- `TROWEXPANDSUB`: `docs/isa/TROWEXPANDSUB.md`
- `TCOLSUM`: `docs/isa/TCOLSUM.md`
- `TCOLMAX`: `docs/isa/TCOLMAX.md`
- `TCOLMIN`: `docs/isa/TCOLMIN.md`

## Memory (GM <-> Tile)
- `TLOAD`: `docs/isa/TLOAD.md`
- `TSTORE`: `docs/isa/TSTORE.md`
- `TSTORE_FP`: `docs/isa/TSTORE_FP.md`

## Matrix Multiply
- `TMATMUL`: `docs/isa/TMATMUL.md`
- `TMATMUL_ACC`: `docs/isa/TMATMUL_ACC.md`
- `TMATMUL_BIAS`: `docs/isa/TMATMUL_BIAS.md`

## Data Movement / Layout
- `TMOV`: `docs/isa/TMOV.md`
- `TMOV_FP`: `docs/isa/TMOV_FP.md`
- `TTRANS`: `docs/isa/TTRANS.md`
- `TEXTRACT`: `docs/isa/TEXTRACT.md`
- `TASSIGN`: `docs/isa/TASSIGN.md`

## Complex
- `TCI`: `docs/isa/TCI.md`
- `TGATHER`: `docs/isa/TGATHER.md`
- `TGATHERB`: `docs/isa/TGATHERB.md`
- `TSCATTER`: `docs/isa/TSCATTER.md`
- `TSORT32`: `docs/isa/TSORT32.md`
- `TMRGSORT`: `docs/isa/TMRGSORT.md`
- `TPARTADD`: `docs/isa/TPARTADD.md`
- `TPARTMAX`: `docs/isa/TPARTMAX.md`
- `TPARTMIN`: `docs/isa/TPARTMIN.md`

## Synchronization
- `TSYNC`: `docs/isa/TSYNC.md`
