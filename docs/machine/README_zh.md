# PTO 抽象机器（Abstract Machine）

本目录定义 PTO ISA 与 PTO Tile Lib 使用的抽象执行模型。该模型刻意站在“写代码的人”的视角：描述正确程序可以依赖的行为假设，而不要求读者理解每一代设备的所有微架构细节。

## 文档

- PTO 机器模型（core/device/host）：[`docs/machine/abstract-machine.md`](abstract-machine.md)

## 与其他文档的关系

- 数据模型与编程模型：
  - Tiles：[`docs/coding/Tile.md`](../coding/Tile.md)
  - 全局内存张量：[`docs/coding/GlobalTensor.md`](../coding/GlobalTensor.md)
  - 事件与同步：[`docs/coding/Event.md`](../coding/Event.md)
  - 标量值与枚举：[`docs/coding/Scalar.md`](../coding/Scalar.md)
- 指令参考：[`docs/isa/README.md`](../isa/README.md)
