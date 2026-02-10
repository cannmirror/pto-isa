# TFILLPAD_EXPAND


## Tile Operation Diagram

![TFILLPAD_EXPAND tile operation](../figures/isa/TFILLPAD_EXPAND.svg)

## Introduction

Expand fill/pad variant of TFILLPAD (allows dst to be larger than src; implementation-defined).

## See also

- TFILLPAD overview and constraints: `docs/isa/TFILLPAD.md`.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

### IR Level 1 (SSA)

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
