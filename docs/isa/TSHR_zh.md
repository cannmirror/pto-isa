# TSHR

## 指令示意图

![TSHR tile operation](../figures/isa/TSHR.svg)

## 简介

两个 Tile 的逐元素右移。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \gg \mathrm{src1}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`.

同步形式：

```text
%dst = tshr %src0, %src1 : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);
```

## 约束

- Intended for integral element types.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT x, sh, out;
  TSHR(out, x, sh);
}
```
