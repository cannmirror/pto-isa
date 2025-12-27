# TCVT

## Introduction

Elementwise type conversion with a specified rounding mode.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) $$

where `rmode` is a rounding policy (see `pto::RoundMode`).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

| Rounding Mode | Description                                                                                      |
|---------------|--------------------------------------------------------------------------------------------------|
|     NONE      | It indicates RINT mode when there is a loss of precision in the conversion, and no rounding when there is no loss of precision involved                                                                             |
|     RINT      | rint, rounded to 50% double rounding                                                             |
|     ROUND     | round, rounding                                                                                  |
|     FLOOR     | floor, and the negative is infinitely rounded                                                    |
|     CEIL      | ceil, to the right infinite rounding                                                             |
|     TRUNC     | trunc, rounded to zero                                                                           |
|     ODD       | Von Neumann rounding, the nearest neighbour is rounded to the odd number                         |
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, WaitEvents&... events);
```

## Constraints

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- The conversion `(src element type) -> (dst element type)` must be supported by the target for the given `RoundMode`.
- **Implementation notes (A2A3/A5)**:
  - `TCVT_IMPL` does not enforce additional `static_assert`/`PTO_ASSERT` checks on the type pair; unsupported conversions are target-defined.

Supported data types:
| src type/dst type | float    | half     | bfloat16_t | int32_t  | int16_t  | int8_t   | uint32_t | uint16_t | uint8_t  |
|-------------------|----------|----------|------------|----------|----------|----------|----------|----------|----------|
|  float            | A2/A3/A5 | A2/A3/A5 |  A2/A3/A5  | A2/A3/A5 | A2/A3/A5 |          |          |          |          |
|  half             | A2/A3/A5 |          |            | A2/A3/A5 | A2/A3/A5 | A2/A3/A5 |          |          | A2/A3/A5 |
|  bfloat16_t       | A2/A3/A5 | A5       |            | A2/A3/A5 |          |          |          |          |          |
|  int32_t          | A2/A3/A5 | A2/A3    |            |          | A2/A3/A5 |          |          | A5       |          |
|  int16_t          | A2/A3/A5 | A2/A3/A5 |            | A5       |          |          | A5       |          | A5       |
|  int8_t           | A5       | A2/A3    |            | A5       | A5       |          |          |          |          |
|  uint32_t         |          |          |            |          |          |          |          |          |          |
|  uint16_t         |          |          |            |          |          |          |          |          |          |
|  uint8_t          | A5       | A2/A3    |            |          |          |          | A5       | A5       |          |

The RoundMode limit is shown in the following table:
|      src      |      dst     | NONE | RINT | FLOOR | CEIL | ROUND | TRUNC | ODD |
|---------------|--------------|------|------|-------|------|-------|-------|-----|
| float         | float        |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | half         |  Y   |  Y   |   Y   |   Y  |   Y   |   Y   |  Y  |
|               | bfloat16_t   |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | int32_t      |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | int16_t      |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
| half          | float        |  Y   |      |       |      |       |       |     |
|               | int32_t      |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | int16_t      |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | int8_t       |  Y   |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | uint8_t      |  Y   |  Y   |   Y   |   Y  |   Y   |   Y   |     |
| bfloat16_t    | float        |  Y   |      |       |      |       |       |     |
|               | int32_t      |  Y   |  Y   |   Y   |   Y  |   Y   |   Y   |     |
| int32_t       | float        |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | half         |  Y   |      |       |      |       |       |     |
|               | int16_t      |  Y   |      |       |      |       |       |     |
|               | half         |  Y   |      |       |      |       |       |     |
| int16_t       | half         |      |  Y   |   Y   |   Y  |   Y   |   Y   |     |
|               | float        |  Y   |      |       |      |       |       |     |
| uint8_t       | half         |  Y   |      |       |      |       |       |     |
| int8_t        | half         |  Y   |      |       |      |       |       |     |
## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```
