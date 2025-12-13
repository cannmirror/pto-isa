# 固定管线/搬运（Fix Pipe Op）
在不同 Tile 间搬运或做简单变换，常用于喂数、布局调整或结果下沉。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TMOV | 搬运/格式转换（多重载） | `TMOV %Src -> %Dst` | docs/isa/TMOV.md |
| TTRANS | 转置 | `TTRANS %Src -> %Dst` | docs/isa/TTRANS.md |
| TEXTRACT | 子块截取 | `TEXTRACT %Src, row, col -> %Dst` | docs/isa/TEXTRACT.md |

约束要点：目标形状需匹配搬运/截取后的布局；量化/模式模板参数需与硬件一致。
