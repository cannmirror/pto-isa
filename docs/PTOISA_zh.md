# PTO ISA 概述

本文档提供了 PTO Tile 库指令集的高层概述，并作为 `docs/` 目录树的“内容索引”页。

`docs/isa/` 目录下的每条指令参考页均基于 `pto::T*` 函数公开的 C++ 内部 API 编写，并采用统一的结构和符号体系。

## 文档目录

| 领域 | 页面 | 描述 |
|---|---|---|
| 概述 | README_zh.md | PTO ISA 指南入口及导航。 |
| 概述 | PTOISA_zh.md | 本页面（概述 + 完整指令索引）。 |
| 快速开始 | getting-started_zh.md | 构建与运行基础（推荐：从 CPU 开始）。 |
| ISA 参考 | isa/README_zh.md | 每条指令的参考目录索引。 |
| ISA 参考 | isa/conventions.md | 通用符号、操作数、事件和修饰符。 |
| 汇编 (PTO-AS) | grammar/README_zh.md | PTO-AS 文档目录索引。 |
| 汇编 (PTO-AS) | grammar/PTO-AS.md | PTO-AS 语法参考。 |
| 汇编 (PTO-AS) | grammar/conventions.md | 语法符号与约定。 |
| 开发者笔记 | coding/README_zh.md | 扩展与实现说明。 |
| 开发者笔记 | coding/Tile.md | Tile 类型、布局与用法。 |
| 开发者笔记 | coding/GlobalTensor.md | GlobalTensor (GM) 包装器与布局。 |
| 开发者笔记 | coding/Scalar.md | Scalar 包装器与立即数。 |
| 开发者笔记 | coding/Event.md | 事件令牌与同步。 |
| 开发者笔记 | coding/ProgrammingModel.md | PTO-Auto 与 PTO-Manual 概述。 |
| 机器模型 | machine/abstract-machine_zh.md | 抽象执行机器模型。 |
| 权威源 | reference/pto-intrinsics-header.md | C++ 内部 API（权威来源）。 |

## 编程模型

ISA 参考基于以下程序员可见的模型：

*   Tile 与有效区域语义：`docs/coding/Tile.md`
*   全局内存张量模型：`docs/coding/GlobalTensor.md`
*   事件与同步模型：`docs/coding/Event.md`
*   标量值、类型助记符与枚举：`docs/coding/Scalar.md`
*   PTO-Auto 与 PTO-Manual 概述：`docs/coding/ProgrammingModel.md`

关于抽象执行模型（Tile 块如何在核心/设备间调度），请参阅 `docs/machine/abstract-machine.md`。

## 指令分类

| 分类 | 描述 |
|---|---|
| 同步 | 执行顺序原语（事件等待与流水线屏障）。 |
| 手动 / 资源绑定 | 用于将 Tile/资源绑定到实现定义的片上地址的手动模式原语。 |
| 逐元素运算 (Tile-Tile) | 消费两个 Tile (Tile-Tile) 并写入一个 Tile 结果的逐元素运算。 |
| Tile-标量 / Tile-立即数 | 混合 Tile 与标量/立即数值的逐元素运算，以及少量融合模式。 |
| 轴归约 / 扩展 | 沿单个轴进行归约，以及按行或按列复制值的广播/扩展操作。 |
| 内存 (GM <-> Tile) | 全局内存 (GM) 张量与 Tile 之间的传输，包括间接收集/散播。 |
| 矩阵乘法 | GEMM 风格矩阵乘法变体，生成累加器/输出 Tile，可选偏置/累加/缩放。 |
| 数据移动 / 布局变换 | Tile 数据移动与布局变换，如移动、转置、重塑和子 Tile 提取。 |
| 复合运算 | 高级或非规则运算，如序列生成、基于索引的收集/散播、排序和部分运算。 |

## 指令索引（所有 PTO 指令）

本表涵盖了 `include/pto/common/pto_instr.hpp` 公开的所有 PTO 指令，并链接至各指令的参考页面。

| 分类 | 指令 | 描述 |
|---|---|---|
| 同步 | isa/TSYNC.md | 同步 PTO 执行（等待事件或插入每操作流水线屏障）。 |
| 手动 / 资源绑定 | isa/TASSIGN.md | 将 Tile 对象绑定到实现定义的片上地址（手动放置）。 |
| 手动 / 资源绑定 | isa/TSETFMATRIX.md | 为类 IMG2COL 操作设置 FMATRIX 寄存器。 |
| 逐元素运算 (Tile-Tile) | isa/TABS.md | Tile 的逐元素绝对值。 |
| 逐元素运算 (Tile-Tile) | isa/TADD.md | 两个 Tile 的逐元素加法。 |
| 逐元素运算 (Tile-Tile) | isa/TADDC.md | 三元逐元素加法：`src0 + src1 + src2`。 |
| 逐元素运算 (Tile-Tile) | isa/TAND.md | 两个 Tile 的逐元素按位与。 |
| 逐元素运算 (Tile-Tile) | isa/TCMP.md | 比较两个 Tile 并写入一个打包的谓词掩码。 |
| 逐元素运算 (Tile-Tile) | isa/TCVT.md | 带指定舍入模式的逐元素类型转换。 |
| 逐元素运算 (Tile-Tile) | isa/TDIV.md | 两个 Tile 的逐元素除法。 |
| 逐元素运算 (Tile-Tile) | isa/TEXP.md | 逐元素指数运算。 |
| 逐元素运算 (Tile-Tile) | isa/TLOG.md | Tile 的逐元素自然对数。 |
| 逐元素运算 (Tile-Tile) | isa/TMAX.md | 两个 Tile 的逐元素最大值。 |
| 逐元素运算 (Tile-Tile) | isa/TMIN.md | 两个 Tile 的逐元素最小值。 |
| 逐元素运算 (Tile-Tile) | isa/TMUL.md | 两个 Tile 的逐元素乘法。 |
| 逐元素运算 (Tile-Tile) | isa/TNEG.md | Tile 的逐元素取负。 |
| 逐元素运算 (Tile-Tile) | isa/TNOT.md | Tile 的逐元素按位取反。 |
| 逐元素运算 (Tile-Tile) | isa/TOR.md | 两个 Tile 的逐元素按位或。 |
| 逐元素运算 (Tile-Tile) | isa/TPRELU.md | 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。 |
| 逐元素运算 (Tile-Tile) | isa/TRECIP.md | Tile 的逐元素倒数。 |
| 逐元素运算 (Tile-Tile) | isa/TRELU.md | Tile 的逐元素 ReLU。 |
| 逐元素运算 (Tile-Tile) | isa/TREM.md | 两个 Tile 的逐元素余数，余数符号与除数相同。 |
| 逐元素运算 (Tile-Tile) | isa/TFMOD.md | 两个 Tile 的逐元素余数，余数符号与被除数相同。 |
| 逐元素运算 (Tile-Tile) | isa/TRSQRT.md | 逐元素倒数平方根。 |
| 逐元素运算 (Tile-Tile) | isa/TSEL.md | 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。 |
| 逐元素运算 (Tile-Tile) | isa/TSHL.md | 两个 Tile 的逐元素左移。 |
| 逐元素运算 (Tile-Tile) | isa/TSHR.md | 两个 Tile 的逐元素右移。 |
| 逐元素运算 (Tile-Tile) | isa/TSQRT.md | 逐元素平方根。 |
| 逐元素运算 (Tile-Tile) | isa/TSUB.md | 两个 Tile 的逐元素减法。 |
| 逐元素运算 (Tile-Tile) | isa/TSUBC.md | 三元逐元素运算：`src0 - src1 + src2`。 |
| 逐元素运算 (Tile-Tile) | isa/TXOR.md | 两个 Tile 的逐元素按位异或。 |
| Tile-标量 / Tile-立即数 | isa/TADDS.md | Tile 与标量的逐元素加法。 |
| Tile-标量 / Tile-立即数 | isa/TADDSC.md | 与标量和第二个 Tile 的融合逐元素加法：`src0 + scalar + src1`。 |
| Tile-标量 / Tile-立即数 | isa/TANDS.md | Tile 与标量的逐元素按位与。 |
| Tile-标量 / Tile-立即数 | isa/TCMPS.md | 将 Tile 与标量比较并写入逐元素比较结果。 |
| Tile-标量 / Tile-立即数 | isa/TDIVS.md | 与标量的逐元素除法（Tile/标量 或 标量/Tile）。 |
| Tile-标量 / Tile-立即数 | isa/TEXPANDS.md | 将标量广播到目标 Tile 中。 |
| Tile-标量 / Tile-立即数 | isa/TLRELU.md | 带标量斜率的 Leaky ReLU。 |
| Tile-标量 / Tile-立即数 | isa/TMAXS.md | Tile 与标量的逐元素最大值：`max(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | isa/TMINS.md | Tile 与标量的逐元素最小值。 |
| Tile-标量 / Tile-立即数 | isa/TMULS.md | Tile 与标量的逐元素乘法。 |
| Tile-标量 / Tile-立即数 | isa/TORS.md | Tile 与标量的逐元素按位或。 |
| Tile-标量 / Tile-立即数 | isa/TREMS.md | 与标量的逐元素余数：`remainder(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | isa/TREMS.md | 与标量的逐元素余数：`fmod(src, scalar)`。 |
| Tile-标量 / Tile-立即数 | isa/TSELS.md | 使用标量 `selectMode` 在两个源 Tile 中选择一个（全局选择）。 |
| Tile-标量 / Tile-立即数 | isa/TSUBS.md | 从 Tile 中逐元素减去一个标量。 |
| Tile-标量 / Tile-立即数 | isa/TSUBSC.md | 融合逐元素运算：`src0 - scalar + src1`。 |
| Tile-标量 / Tile-立即数 | isa/TXORS.md | Tile 与标量的逐元素按位异或。 |
| Tile-标量 / Tile-立即数 | isa/TSHLS.md | Tile 按标量逐元素左移。 |
| Tile-标量 / Tile-立即数 | isa/TSHRS.md | Tile 按标量逐元素右移。 |
| 轴归约 / 扩展 | isa/TCOLEXPAND.md | 将每个源列的第一个元素广播到目标列中。 |
| 轴归约 / 扩展 | isa/TCOLMAX.md | 通过取行间最大值来归约每一列。 |
| 轴归约 / 扩展 | isa/TCOLMIN.md | 通过取行间最小值来归约每一列。 |
| 轴归约 / 扩展 | isa/TCOLSUM.md | 通过对行求和来归约每一列。 |
| 轴归约 / 扩展 | isa/TROWEXPAND.md | 将每个源行的第一个元素广播到目标行中。 |
| 轴归约 / 扩展 | isa/TROWEXPANDDIV.md | 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | isa/TROWEXPANDMUL.md | 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | isa/TROWEXPANDSUB.md | 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。 |
| 轴归约 / 扩展 | isa/TROWMAX.md | 通过取列间最大值来归约每一行。 |
| 轴归约 / 扩展 | isa/TROWMIN.md | 通过取列间最小值来归约每一行。 |
| 轴归约 / 扩展 | isa/TROWSUM.md | 通过对列求和来归约每一行。 |
| 轴归约 / 扩展 | isa/TCOLEXPANDDIV.md | 列广播除法：将每一列除以一个每列标量向量。 |
| 轴归约 / 扩展 | isa/TCOLEXPANDEXPDIF.md | 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。 |
| 轴归约 / 扩展 | isa/TCOLEXPANDMUL.md | 列广播乘法：将每一列乘以一个每列标量向量。 |
| 轴归约 / 扩展 | isa/TCOLEXPANDSUB.md | 列广播减法：从每一列中减去一个每列标量向量。 |
| 轴归约 / 扩展 | isa/TROWEXPANDADD.md | 行广播加法：加上一个每行标量向量。 |
| 轴归约 / 扩展 | isa/TROWEXPANDEXPDIF.md | 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。 |
| 轴归约 / 扩展 | isa/TROWEXPANDMAX.md | 行广播最大值：与每行标量向量取最大值。 |
| 轴归约 / 扩展 | isa/TROWEXPANDMIN.md | 行广播最小值：与每行标量向量取最小值。 |
| 内存 (GM <-> Tile) | isa/MGATHER.md | 使用逐元素索引从全局内存收集加载元素到 Tile 中。 |
| 内存 (GM <-> Tile) | isa/MSCATTER.md | 使用逐元素索引将 Tile 中的元素散播存储到全局内存。 |
| 内存 (GM <-> Tile) | isa/TLOAD.md | 从 GlobalTensor (GM) 加载数据到 Tile。 |
| 内存 (GM <-> Tile) | isa/TSTORE.md | 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。 |
| 内存 (GM <-> Tile) | isa/TSTORE_FP.md | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。 |
| 内存 (GM <-> Tile) | isa/TPREFETCH.md | 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。 |
| 矩阵乘法 | isa/TMATMUL.md | 矩阵乘法 (GEMM)，生成累加器/输出 Tile。 |
| 矩阵乘法 | isa/TMATMUL_ACC.md | 带累加器输入的矩阵乘法（融合累加）。 |
| 矩阵乘法 | isa/TMATMUL_BIAS.md | 带偏置加法的矩阵乘法。 |
| 矩阵乘法 | isa/TMATMUL_MX.md | 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。 |
| 矩阵乘法 | isa/TGEMV.md | 通用矩阵-向量乘法，生成累加器/输出 Tile。 |
| 矩阵乘法 | isa/TGEMV_ACC.md | 带显式累加器输入/输出 Tile 的 GEMV。 |
| 矩阵乘法 | isa/TGEMV_BIAS.md | 带偏置加法的 GEMV。 |
| 数据移动 / 布局变换 | isa/TEXTRACT.md | 从源 Tile 中提取子 Tile。 |
| 数据移动 / 布局变换 | isa/TMOV.md | 在 Tile 之间移动/复制，可选应用实现定义的转换模式。 |
| 数据移动 / 布局变换 | isa/TMOV_FP.md | 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。 |
| 数据移动 / 布局变换 | isa/TRESHAPE.md | 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。 |
| 数据移动 / 布局变换 | isa/TTRANS.md | 使用实现定义的临时 Tile 进行转置。 |
| 数据移动 / 布局变换 | isa/TEXTRACT_FP.md | 带 fp/缩放 Tile 的提取（向量量化参数）。 |
| 数据移动 / 布局变换 | isa/TFILLPAD.md | 复制 Tile 并在有效区域外使用编译时填充值进行填充。 |
| 数据移动 / 布局变换 | isa/TFILLPAD_EXPAND.md | 填充/填充时允许目标大于源。 |
| 数据移动 / 布局变换 | isa/TFILLPAD_INPLACE.md | 原地填充/填充变体。 |
| 数据移动 / 布局变换 | isa/TIMG2COL.md | 用于类卷积工作负载的图像到列变换。 |
| 数据移动 / 布局变换 | isa/TINSERT.md | 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。 |
| 数据移动 / 布局变换 | isa/TINSERT_FP.md | 带 fp/缩放 Tile 的插入（向量量化参数）。 |
| 复合运算 | isa/TCI.md | 生成连续整数序列到目标 Tile 中。 |
| 复合运算 | isa/TGATHER.md | 使用索引 Tile 或编译时掩码模式来收集/选择元素。 |
| 复合运算 | isa/TGATHERB.md | 使用字节偏移量收集元素。 |
| 复合运算 | isa/TMRGSORT.md | 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。 |
| 复合运算 | isa/TPARTADD.md | 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。 |
| 复合运算 | isa/TPARTMAX.md | 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复合运算 | isa/TPARTMIN.md | 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。 |
| 复合运算 | isa/TSCATTER.md | 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。 |
| 复合运算 | isa/TSORT32.md | 对固定大小的 32 元素块进行排序并生成索引映射。 |
| 复合运算 | isa/TPRINT.md | 调试/打印 Tile 中的元素（实现定义）。 |
| 复合运算 | isa/TQUANT.md | 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。 |
| 复合运算 | isa/TTRI.md | 生成三角（下/上）掩码 Tile。 |