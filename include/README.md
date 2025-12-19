# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See `include/pto/README.md` for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu)

## Related Docs

- ISA guide: `docs/README.md`
- Getting started: `docs/getting-started.md`

## PTO Instruction Implementation Status (CPU / A2 / A3)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend).
- **A2 / A3**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **TODO** means the instruction is part of the public API but the backend implementation is not available yet.

| Instruction | CPU | A2 | A3 |
|---|---:|---:|---:|
| [`TASSIGN`](../docs/isa/TASSIGN.md) | Yes | Yes | Yes |
| [`TSYNC`](../docs/isa/TSYNC.md) | Yes | Yes | Yes |
| [`TADD`](../docs/isa/TADD.md) | Yes | Yes | Yes |
| [`TABS`](../docs/isa/TABS.md) | Yes | Yes | Yes |
| [`TSUB`](../docs/isa/TSUB.md) | Yes | Yes | Yes |
| [`TMUL`](../docs/isa/TMUL.md) | Yes | Yes | Yes |
| [`TMIN`](../docs/isa/TMIN.md) | Yes | Yes | Yes |
| [`TMAX`](../docs/isa/TMAX.md) | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | Yes | Yes | Yes |
| [`TLOAD`](../docs/isa/TLOAD.md) | Yes | Yes | Yes |
| [`TCMPS`](../docs/isa/TCMPS.md) | Yes | Yes | Yes |
| [`TCMP`](../docs/isa/TCMP.md) | Yes | Yes | Yes |
| [`TSTORE`](../docs/isa/TSTORE.md) | Yes | Yes | Yes |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | Yes | Yes | Yes |
| [`TDIV`](../docs/isa/TDIV.md) | Yes | Yes | Yes |
| [`TREM`](../docs/isa/TREM.md) | Yes | TODO | TODO |
| [`TSHL`](../docs/isa/TSHL.md) | Yes | TODO | TODO |
| [`TSHR`](../docs/isa/TSHR.md) | Yes | TODO | TODO |
| [`TAND`](../docs/isa/TAND.md) | Yes | TODO | TODO |
| [`TOR`](../docs/isa/TOR.md) | Yes | TODO | TODO |
| [`TXOR`](../docs/isa/TXOR.md) | Yes | TODO | TODO |
| [`TLOG`](../docs/isa/TLOG.md) | Yes | Yes | Yes |
| [`TNEG`](../docs/isa/TNEG.md) | Yes | TODO | TODO |
| [`TNOT`](../docs/isa/TNOT.md) | Yes | TODO | TODO |
| [`TRECIP`](../docs/isa/TRECIP.md) | Yes | Yes | Yes |
| [`TRELU`](../docs/isa/TRELU.md) | Yes | TODO | TODO |
| [`TPRELU`](../docs/isa/TPRELU.md) | Yes | TODO | TODO |
| [`TADDC`](../docs/isa/TADDC.md) | Yes | TODO | TODO |
| [`TSUBC`](../docs/isa/TSUBC.md) | Yes | TODO | TODO |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | Yes | Yes | Yes |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | Yes | Yes | Yes |
| [`TSORT32`](../docs/isa/TSORT32.md) | Yes | Yes | Yes |
| [`TGATHER`](../docs/isa/TGATHER.md) | Yes | Yes | Yes |
| [`TCI`](../docs/isa/TCI.md) | Yes | Yes | Yes |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | Yes | Yes | Yes |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | Yes | Yes | Yes |
| [`TCVT`](../docs/isa/TCVT.md) | Yes | Yes | Yes |
| [`TMOV`](../docs/isa/TMOV.md) | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | Yes | Yes | Yes |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | Yes | Yes | Yes |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | Yes | Yes | Yes |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | Yes | Yes | Yes |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | Yes | Yes | Yes |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | Yes | Yes | Yes |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | Yes | Yes | Yes |
| [`TSELS`](../docs/isa/TSELS.md) | Yes | Yes | Yes |
| [`TSEL`](../docs/isa/TSEL.md) | Yes | Yes | Yes |
| [`TTRANS`](../docs/isa/TTRANS.md) | Yes | Yes | Yes |
| [`TMINS`](../docs/isa/TMINS.md) | Yes | Yes | Yes |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | Yes | Yes | Yes |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | Yes | Yes | Yes |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL.md) | Yes | Yes | Yes |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | Yes | Yes | Yes |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | Yes | Yes | Yes |
| [`TSQRT`](../docs/isa/TSQRT.md) | Yes | Yes | Yes |
| [`TEXP`](../docs/isa/TEXP.md) | Yes | Yes | Yes |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | Yes | Yes | Yes |
| [`TADDS`](../docs/isa/TADDS.md) | Yes | Yes | Yes |
| [`TSUBS`](../docs/isa/TSUBS.md) | Yes | TODO | TODO |
| [`TDIVS`](../docs/isa/TDIVS.md) | Yes | Yes | Yes |
| [`TMULS`](../docs/isa/TMULS.md) | Yes | Yes | Yes |
| [`TREMS`](../docs/isa/TREMS.md) | Yes | TODO | TODO |
| [`TMAXS`](../docs/isa/TMAXS.md) | Yes | TODO | TODO |
| [`TANDS`](../docs/isa/TANDS.md) | Yes | TODO | TODO |
| [`TORS`](../docs/isa/TORS.md) | Yes | TODO | TODO |
| [`TXORS`](../docs/isa/TXORS.md) | Yes | TODO | TODO |
| [`TLRELU`](../docs/isa/TLRELU.md) | Yes | TODO | TODO |
| [`TADDSC`](../docs/isa/TADDSC.md) | Yes | TODO | TODO |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | Yes | TODO | TODO |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | Yes | Yes | Yes |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | Yes | TODO | TODO |
| [`MGATHER`](../docs/isa/MGATHER.md) | Yes | TODO | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | Yes | TODO | TODO |
