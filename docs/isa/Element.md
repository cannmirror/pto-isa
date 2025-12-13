# 逐元素操作（Element Op）
涵盖 Tile 间逐元素算术、选择、类型转换等操作，接口来自 `include/pto/common/pto_instr.hpp`。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TADD | 逐元素加 | `TADD %Src0, %Src1 -> %Dst` | docs/isa/TADD.md |
| TSUB | 逐元素减 | `TSUB %Src0, %Src1 -> %Dst` | docs/isa/TSUB.md |
| TMUL | 逐元素乘 | `TMUL %Src0, %Src1 -> %Dst` | docs/isa/TMUL.md |
| TDIV | 逐元素除 | `TDIV %Src0, %Src1 -> %Dst` | docs/isa/TDIV.md |
| TMIN | 逐元素最小 | `TMIN %Src0, %Src1 -> %Dst` | docs/isa/TMIN.md |
| TMAX | 逐元素最大 | `TMAX %Src0, %Src1 -> %Dst` | docs/isa/TMAX.md |
| TEXP | 指数 | `TEXP %Src -> %Dst` | docs/isa/TEXP.md |
| TSQRT | 平方根 | `TSQRT %Src -> %Dst` | docs/isa/TSQRT.md |
| TRSQRT | 1/sqrt | `TRSQRT %Src -> %Dst` | docs/isa/TRSQRT.md |
| TEXPANDS | 标量扩展 Tile | `TEXPANDS %R -> %Dst` | docs/isa/TEXPANDS.md |
| TSEL | 掩码选择 | `TSEL %Mask, %Src0, %Src1 -> %Dst` | docs/isa/TSEL.md |
| TSELS | 模式选择 | `TSELS %Src0, %Src1, selectMode -> %Dst` | docs/isa/TSELS.md |
| TCVT | 类型转换 | `TCVT.rmode %Src -> %Dst` | docs/isa/TCVT.md |

约束要点：源/目标 Tile 形状一致；数据类型一致（除 TCVT 外）；掩码/模式需匹配有效区域；满足硬件对齐要求。
