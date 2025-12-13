# Tile 操作数

## 说明

Tile 寄存器是一种用于存储向量或矩阵等一、二维数据的片上寄存器，也是 PTO 指令间数据传递的**主要媒介**。

每个 Tile 寄存器的容量（Size）必须满足：

* **容量是 512 字节（矩阵运算最小分形） 的整数倍**；
* **容量范围为 [512B, 32KB]**，不允许超出该范围。

> 直观理解：Tile 就是一块固定大小的「数据集装箱」，所有算子之间都是以 Tile 为单位搬运和计算。

---

## 使用声明

在 PTO C++ 编程模型中，Tile Register 统一通过 `Tile` 模板类型进行声明：

```cpp
Tile<TileType, ElemType, Row, Col, BLayout, RowValid, ColValid, SLayout, SLayoutSize, PadValue> name;
```

其中各模板参数含义如下：

* **`name`**：Tile 寄存器的变量名（例如 `t0`、`attn_q` 等）。
* **`TileType`**：Tile 的类型信息，对应物理位置/用途。
  枚举：`TileType::Vec` / `TileType::Mat` / `TileType::Left` / `TileType::Right` / `TileType::Acc` 等。
  详见 [TileType](./type.md) 小节。
* **`ElemType`**：元素数据类型，例如 `float`、`half`、`fp8`、`int32_t`、`int16_t` 等。
  详见 [DataType](../../datatype/intro.md) 章节。
* **`Row, Col`**：Tile 的静态形状（行数 / 列数），必须为**编译期常量**。
  详见 [Shape](./shape.md) 小节。
* **`BLayout`**（可选）：大分型布局，描述 Tile 在逻辑矩阵上的排布方式。
  枚举：`BLayout::RowMajor` / `BLayout::ColMajor`，默认 `RowMajor`。
  详见 [Layout](./layout.md) 小节。
* **`RowValid, ColValid`**（可选）：动态 Mask 信息，支持**静态或动态**有效尺寸。
  默认：`RowValid = Row`，`ColValid = Col`。
  若设为 `-1`，则表示该维度采用**运行时动态有效长度**。
  详见 [Shape](./shape.md) 小节。
* **`SLayout`**（可选）：小分型布局，用于描述 Tile 内部分块（如 Zz / Nz）。
  枚举：`SLayout::NoneBox` / `SLayout::RowMajor` / `SLayout::ColMajor`，默认 `NoneBox`。
  详见 [Layout](./layout.md) 小节。
* **`SLayoutSize`**（可选）：小分型大小（单位：字节），默认 `512`。
* **`PadValue`**（可选）：超出有效区域的填充值。
  枚举：`PadValue::Null` / `PadValue::Zero` / `PadValue::Max` / `PadValue::Min`，默认 `Null`。

---

## 定义示例

### 静态 Mask 示例

```cpp
// 128 × 256 的 Vec Tile，静态有效区域为 127 × 127，超出部分按 0 填充
using TileT0 = Tile<
    TileType::Vec,       // TileType
    float,               // ElemType
    128, 256,            // Row, Col
    BLayout::RowMajor,   // BLayout
    127, 127,            // RowValid, ColValid（静态 Mask）
    SLayout::NoneBox,    // SLayout
    512,                 // SLayoutSize
    PadValue::Zero       // PadValue
>;
TileT0 t0;
```

### 动态 Mask 示例

```cpp
// 128 × 256 的 Vec Tile，行方向有效长度运行时决定，列方向静态为 127
using TileT1 = Tile<
    TileType::Vec,
    float,
    128, 256,
    BLayout::RowMajor,
    -1, 127              // RowValid = -1 表示动态 Mask
>;
TileT1 t1( /*row_valid=*/120, /*col_valid=*/127 );
```

当 `RowValid` 或 `ColValid` 在模板参数中设置为 `-1` 时：

* 表示该维度的有效范围由**构造函数**或**运行时接口**传入；
* 编译期依然以 `Row` / `Col` 作为 Tile 的静态总容量，用于对齐与指令合法性检查。

---

## Tile 的约束

Tile 寄存器在定义与使用时需要满足以下约束：

1. **容量约束**

   * Tile 总容量 `Row * Col * sizeof(ElemType)` 必须在 **512B ~ 32KB** 范围内；
   * 且必须是 **512B 的整数倍**。

2. **形状（Shape）约束**

   * Tile 最多支持二维形状：`Row × Col`；
   * `Col` 方向必须保证是 **32 字节对齐**（即 `Col * sizeof(ElemType)` 为 32 的倍数）；
   * `Row` 与 `Col` 都为编译期常量（静态 shape）。

3. **数据粒度约束**

   * Tile 是 PTO 指令间数据交换的**最小颗粒度单位**；
   * 单条指令的输入/输出始终以 Tile 为单位，**不能在指令层面对同一 Tile 做「半块读 / 半块写」的拆分**（拆分需通过多个 Tile 或 Mask 表达）。

4. **静态 Shape 与动态 Mask**

   * Tile 的 `Row` / `Col` 在定义后不可修改（静态 Shape）；
   * `RowValid` / `ColValid` 可以静态或动态指定，用于表达实际参与计算的数据范围；
   * 动态 Mask 必须满足：`0 < RowValid ≤ Row`，`0 < ColValid ≤ Col`。

5. **写入独占性**

   * 每条指令对其输出 Tile Register 具有**独占写权限**：

     * **不允许多条指令同时写同一个 Tile 寄存器**；
     * 当前指令尚未提交前，其它指令不可读取同一输出 Tile 的内容（避免读写冲突）。
   * 换言之，Tile 写入在指令粒度上是**原子**的。

6. **生命周期与复用**

   * Tile 类型（`TileType` / `ElemType` / `Row` / `Col` / 布局等）在定义后不可变更；
   * 可以在同一 Tile 寄存器上多次复用（作为不同指令的输出），但每次使用必须遵守前述写入独占与同步规则；
   * 编译器或运行时可以将不同逻辑 Tile 映射到同一片上 Buffer，实现物理复用，但对用户而言这是透明的。
