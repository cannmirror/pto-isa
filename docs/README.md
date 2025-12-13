# PTO ISA 指南

PTO ISA 为 Ascend AI Core 的 Tile 级指令集合，提供加载、计算、转换、访存、通信等能力，帮助在 NPU 上快速实现高性能算子。本文档概览指令分类、命名约定以及各子文档导航。

## 命名与约定
- `Tile`：片上寄存器级数据块，含 MatTile/LeftTile/RightTile/BiasTile/AccTile/VecTile 等。
- `GlobalTensor`：全局内存（GM）上的张量，`TLOAD`/`TSTORE` 在 GM 与 Tile 间搬运。
- `%R`：标量/立即数寄存器；`cmpMode`、`rmode` 等为指令修饰符。
- 形状/对齐由编译期和运行时断言共同保障，错误时指令会主动报错。

## 目录导航
- ISA 详解：`docs/isa/`（按分类拆分；单指令详解见 `docs/isa/<Instr>.md`）
  - 逐元素：`docs/isa/Element.md`
  - 标量：`docs/isa/TileScalar.md`
  - 按轴归约：`docs/isa/Axis.md`
  - 矩阵乘：`docs/isa/Matmul.md`
  - 固定管线：`docs/isa/FixPipe.md`
  - 访存：`docs/isa/Mem.md`
  - 复杂操作：`docs/isa/Complex.md`
  - 通信：`docs/isa/Communication.md`
  - 自定义/手动：`docs/isa/Custom.md`、`docs/isa/Manual.md`
- 代码示例与实现细节：`docs/coding/`

## 指令总览

### 逐元素（Element）
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :------- | :---------------- | :------------------------------ | :----------------------------- |
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

### Tile-Scalar
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :--------- | :---------------- | :------------------------------ | :----------------------------- |
| TADDS | Tile + 标量 | `TADDS %Src, %R -> %Dst` | docs/isa/TADDS.md |
| TDIVS | Tile ÷ 标量/标量 ÷ Tile | `TDIVS %Src, %R -> %Dst`；`TDIVS %R, %Src -> %Dst` | docs/isa/TDIVS.md |
| TMULS | Tile × 标量 | `TMULS %Src, %R -> %Dst` | docs/isa/TMULS.md |
| TMINS | Tile 与标量逐元素最小 | `TMINS %Src, %R -> %Dst` | docs/isa/TMINS.md |
| TCMPS | Tile 与标量比较掩码 | `TCMPS.cmpMode %Src, %R -> %Dst` | docs/isa/TCMPS.md |

### 按轴归约
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :----------- | :---------------- | :------------------------------ | :----------------------------- |
| TROWSUM | 行求和 | `TROWSUM %Src -> %Dst` | docs/isa/TROWSUM.md |
| TROWMAX | 行最大 | `TROWMAX %Src -> %Dst` | docs/isa/TROWMAX.md |
| TROWMIN | 行最小 | `TROWMIN %Src -> %Dst` | docs/isa/TROWMIN.md |
| TROWEXPAND | 行首元素广播整行 | `TROWEXPAND %Src -> %Dst` | docs/isa/TROWEXPAND.md |
| TCOLSUM | 列求和 | `TCOLSUM %Src -> %Dst` | docs/isa/TCOLSUM.md |
| TCOLMAX | 列最大 | `TCOLMAX %Src -> %Dst` | docs/isa/TCOLMAX.md |
| TCOLMIN | 列最小 | `TCOLMIN %Src -> %Dst` | docs/isa/TCOLMIN.md |

### 矩阵乘
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :------------- | :-------------------- | :------------------------------------------ | :------------------------------- |
| TMATMUL | 基础 GEMM | `TMATMUL %Left, %Right -> %Acc` | docs/isa/TMATMUL.md |
| TMATMUL_ACC | GEMM 累加 | `TMATMUL_ACC %Left, %Right, %Acc -> %Acc` | docs/isa/TMATMUL_ACC.md |
| TMATMUL_BIAS | GEMM + Bias | `TMATMUL_BIAS %Left, %Right, %Bias -> %Acc` | docs/isa/TMATMUL_BIAS.md |

### 固定管线/搬运
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :----------- | :------------------ | :------------------------------------------ | :--------------------------- |
| TMOV | Tile 间搬运/转换（可选模式/量化） | `TMOV %Src -> %Dst` | docs/isa/TMOV.md |
| TTRANS | 转置 | `TTRANS %Src -> %Dst` | docs/isa/TTRANS.md |
| TEXTRACT | 子块截取 | `TEXTRACT %Src, row, col -> %Dst` | docs/isa/TEXTRACT.md |

### 访存（GM ↔ Tile）
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :------------ | :------------------ | :------------------------------------------ | :--------------------------- |
| TLOAD | GM → Tile 加载 | `TLOAD %Global -> %Tile` | docs/isa/TLOAD.md |
| TSTORE | Tile → GM 存储（可原子/量化） | `TSTORE %Tile -> %Global` | docs/isa/TSTORE.md |

### 复杂操作
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :---------- | :------------------------ | :-------------------------------------------- | :---------------------------- |
| TCI | 连续整数序列 | `TCI.asc/desc imm -> %Dst` | docs/isa/TCI.md |
| TGATHER | Tile 内聚集（索引/模式） | `TGATHER %Src0, %Src1 -> %Dst` 或 `TGATHER.maskPattern %Src -> %Dst` | docs/isa/TGATHER.md |
| TGATHERB | Tile 偏移聚集 | `TGATHERB %Src, %Offset -> %Dst` | docs/isa/TGATHERB.md |
| TSORT32 | 32 元素块排序 | `TSORT32 %Src, %Idx -> %Dst` | docs/isa/TSORT32.md |
| TMRGSORT | 归并排序（多路/块） | `TMRGSORT %Src... -> %Dst` | docs/isa/TMRGSORT.md |
| TPARTADD | 分段掩码加法 | `TPARTADD %Src0, %Src1 -> %Dst` | docs/isa/TPARTADD.md |
| TPARTMAX | 分段掩码最大 | `TPARTMAX %Src0, %Src1 -> %Dst` | docs/isa/TPARTMAX.md |
| TPARTMIN | 分段掩码最小 | `TPARTMIN %Src0, %Src1 -> %Dst` | docs/isa/TPARTMIN.md |

### 通信


### 手动/资源
| PTO 指令 | 功能 | 接口示例 | 文档 |
| :-------- | :--------------------- | :-------------------------------- | :----------------------------- |
| TASSIGN | 显式绑定 Tile 缓冲地址 | `TASSIGN %Tile, %addr` | docs/isa/TASSIGN.md |
