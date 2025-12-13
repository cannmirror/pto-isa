# 逐元素与标量操作（Tile-Scalar Op）
Tile 与标量寄存器 `%R` 的逐元素计算，接口来自 `include/pto/common/pto_instr.hpp`。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TADDS | Tile + 标量 | `TADDS %Src, %R -> %Dst` | docs/isa/TADDS.md |
| TDIVS | Tile ÷ 标量 / 标量 ÷ Tile | `TDIVS %Src, %R -> %Dst`；`TDIVS %R, %Src -> %Dst` | docs/isa/TDIVS.md |
| TMULS | Tile × 标量 | `TMULS %Src, %R -> %Dst` | docs/isa/TMULS.md |
| TMINS | Tile 与标量逐元素最小 | `TMINS %Src, %R -> %Dst` | docs/isa/TMINS.md |
| TCMPS | Tile 与标量比较掩码 | `TCMPS.cmpMode %Src, %R -> %Dst` | docs/isa/TCMPS.md |

约束要点：Tile 数据类型需与标量兼容；形状保持一致；TDIVS 需注意操作数顺序差异。
