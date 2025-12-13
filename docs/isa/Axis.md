# 按轴归约/扩展（Axis Op）
行/列维度的归约与广播，接口来自 `include/pto/common/pto_instr.hpp`。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TROWSUM | 行求和 | `TROWSUM %Src -> %Dst` | docs/isa/TROWSUM.md |
| TROWMAX | 行最大 | `TROWMAX %Src -> %Dst` | docs/isa/TROWMAX.md |
| TROWMIN | 行最小 | `TROWMIN %Src -> %Dst` | docs/isa/TROWMIN.md |
| TROWEXPAND | 行首元素广播整行 | `TROWEXPAND %Src -> %Dst` | docs/isa/TROWEXPAND.md |
| TCOLSUM | 列求和 | `TCOLSUM %Src -> %Dst` | docs/isa/TCOLSUM.md |
| TCOLMAX | 列最大 | `TCOLMAX %Src -> %Dst` | docs/isa/TCOLMAX.md |
| TCOLMIN | 列最小 | `TCOLMIN %Src -> %Dst` | docs/isa/TCOLMIN.md |

约束要点：源 Tile 形状需满足行/列归约；必要时提供 tmp（如 TROWSUM/TCOLSUM）；遵循对齐与掩码约束。
