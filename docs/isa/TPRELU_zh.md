# TPRELU

## 指令示意图

![TPRELU tile operation](../figures/isa/TPRELU.svg)

## 简介

带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}) $$

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`.

同步形式：

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);
```

## 约束

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- Temporary space is required by A3 for calculation, while not used by A5.
- For A3, 2 source Tile, destination Tile, temporary space must in different memory range without overlapping.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, slope, out, tmp;
  TPRELU(out, x, slope, tmp);
}
```
