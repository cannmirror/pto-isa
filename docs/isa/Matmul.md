# 矩阵乘（MATMUL Op）

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TMATMUL | 基础 GEMM | `TMATMUL %Left, %Right -> %Acc` | docs/isa/TMATMUL.md |
| TMATMUL_ACC | GEMM 累加 | `TMATMUL_ACC %Left, %Right, %Acc -> %Acc` | docs/isa/TMATMUL_ACC.md |
| TMATMUL_BIAS | GEMM + Bias | `TMATMUL_BIAS %Left, %Right, %Bias -> %Acc` | docs/isa/TMATMUL_BIAS.md |

约束要点：Left/Right 维度需对齐；Bias/Acc 形状需匹配；类型需符合硬件支持。
