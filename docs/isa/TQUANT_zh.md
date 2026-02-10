# TQUANT

## 指令示意图

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## 简介

量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`.

### IR Level 1（SSA）

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TQUANT(TileDataSrc &src, TileDataExp &exp, TileDataOut &dst,
                            TileDataMax &max, TileDataSrc &scaling, WaitEvents&... events);
```

## 约束

- This instruction is currently implemented for specific targets (see `include/pto/npu/*/TQuant.hpp`).
- Input type requirements and output tile types are mode/target-dependent.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
