# TSETFMATRIX


## Tile Operation Diagram

![TSETFMATRIX tile operation](../figures/isa/TSETFMATRIX.svg)

## Introduction

Set the FMATRIX register(s) used by IMG2COL-like operations from an `Img2colTileConfig` (target/implementation-defined).

## See also

- IMG2COL instruction: `docs/isa/TIMG2COL.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents&... events);
```

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

### IR Level 1 (SSA)

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### IR Level 2 (DPS)

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```
## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
