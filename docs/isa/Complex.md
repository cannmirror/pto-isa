# 复杂操作（Complex Op）
排序、聚集、分段计算等高级指令。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TCI | 连续整数序列 | `TCI.asc/desc imm -> %Dst` | docs/isa/TCI.md |
| TGATHER | 索引/模式聚集 | `TGATHER %Src0, %Src1 -> %Dst` 或 `TGATHER.maskPattern %Src -> %Dst` | docs/isa/TGATHER.md |
| TGATHERB | 按偏移聚集 | `TGATHERB %Src, %Offset -> %Dst` | docs/isa/TGATHERB.md |
| TSORT32 | 32 元素块排序 | `TSORT32 %Src, %Idx -> %Dst` | docs/isa/TSORT32.md |
| TMRGSORT | 归并排序（多路/块） | `TMRGSORT %Src... -> %Dst` | docs/isa/TMRGSORT.md |
| TPARTADD | 分段掩码加法 | `TPARTADD %Src0, %Src1 -> %Dst` | docs/isa/TPARTADD.md |
| TPARTMAX | 分段掩码最大 | `TPARTMAX %Src0, %Src1 -> %Dst` | docs/isa/TPARTMAX.md |
| TPARTMIN | 分段掩码最小 | `TPARTMIN %Src0, %Src1 -> %Dst` | docs/isa/TPARTMIN.md |

约束要点：索引/掩码需合法；归并/排序需满足块长要求；分段掩码与数据布局一致。
