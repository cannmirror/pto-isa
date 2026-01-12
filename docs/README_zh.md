<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA 指南

这里是 PTO Tile Lib 文档入口，介绍 PTO ISA（指令集架构）的命名/符号约定，以及如何查阅“每条指令一页”的参考手册。

## 命名与符号

- **Tile**：小张量的基础数据类型（例如 `MatTile`、`LeftTile`、`RightTile`、`BiasTile`、`AccumulationTile`、`VecTile`）。
- **GlobalTensor**：存放在全局内存（GM）中的张量；`TLOAD`/`TSTORE` 用于在 GM 与 Tile 之间搬运数据。
- **`%R`**：标量/立即数寄存器；例如 `cmpMode`、`rmode` 等字段属于指令修饰符（modifier）。
- **形状与对齐**：通过编译期约束与运行期断言共同约束；不合法的使用应尽快失败（fail fast）。

## 从哪里开始

- ISA 总览：[`docs/PTOISA.md`](PTOISA.md)
- 指令索引：[`docs/isa/README.md`](isa/README.md)
- 通用约定：[`docs/isa/conventions.md`](isa/conventions.md)
- PTO 汇编语法（PTO-AS）：[`docs/grammar/PTO-AS.md`](grammar/PTO-AS.md)
- 入门指南（建议先跑 CPU 仿真）：[`docs/getting-started.md`](getting-started.md)
- 实现与扩展说明：[`docs/coding/README.md`](coding/README.md)
- Kernel 示例（偏 NPU）：[`kernels/README.md`](../kernels/README.md)

## 文档组织

- `docs/isa/`：指令参考（每条指令一页，以及分类索引）
- `docs/grammar/`：PTO 汇编语法与规范（PTO-AS）
- `docs/coding/`：扩展 PTO Tile Lib 的开发者说明
