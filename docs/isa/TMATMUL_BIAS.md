# TMATMUL_BIAS

## 说明
**矩阵乘加 Bias（MatMul Bias）**

- 功能：GEMM + Bias

$$ \mathrm{C} = \mathrm{A} \times \mathrm{B} + \mathrm{Bias} $$

---

## 汇编语法
```asm
TMATMUL_BIAS %Left, %Right, %Bias , -> %Acc
```

### 汇编符号说明
- `%SrcTile/%SrcTile0/%SrcTile1`：输入 Tile（数量与指令匹配）。
- `%DstTile`：输出 Tile。
- `%R`：标量立即数/寄存器（仅标量类指令使用）。
- `cmpMode/rmode/selectMode`：模式修饰或参数（具体含义见 C++ 接口与实现约束）。

---

## C++ Intrinsic 接口
```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INST void TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData);
```

### 参数说明
| 参数 | 含义 |
| ------ | ----------------------------------------- |
| cMatrix | 输出/累加矩阵 Tile |
| aMatrix | 左矩阵 Tile |
| bMatrix | 右矩阵 Tile |
| biasData | Bias Tile |

---

## 语义说明
- 仅对有效区域（由 `Tile::GetValidRow()` / `Tile::GetValidCol()` 决定）内的元素生效。
- 超出有效区域（被 Mask 掉）的元素不参与计算，其结果由实现/先前数据决定。
- 输入/输出 Tile 的形状、布局、数据类型需要满足实现约束。

---

## 指令约束
### 通用约束
1. **形状与有效范围**：`RowValid/ColValid` 不得超过静态 `Rows/Cols`。
2. **对齐与布局**：Tile 模板定义中已包含对齐/布局静态检查（例如 32B 对齐与 Box 布局整除约束）。
3. **实现差异**：不同 SOC/实现（A2A3/A5/CPU_SIM）可能有不同的数据类型与 TileType 限制。

### 实现检查（A2A3）
- 编译期约束：
  - No supported bias data type.
  - TileBias must be single row.

### 实现检查（A5）
- 编译期约束：
  - No supported bias data type.
  - Non-conforming bias fractal.

---

## 编程示例
### PTO Auto 写法
```cpp
#include "pto/common/pto_instr.hpp"
#include "pto/common/pto_tile.hpp"

using namespace pto;
template <typename T>
void example() {
  // 注意：实际 Matmul TileType/布局需与实现约束匹配
  using Left = TileLeft<T, 16, 16>;
  using Right = TileRight<T, 16, 16>;
  using Acc = TileAcc<float, 16, 16>;  // Acc 常用更高精度，按实际实现选择

  Left left;
  Right right;
  Acc acc;

  // 典型流程：TLOAD/布局变换 -> TMATMUL -> TSTORE
  // TMATMUL(acc, left, right);
}
```

### PTO Manual 写法（可选）
- 若启用手动模式并需要显式分配片上地址，可先使用 `TASSIGN` 绑定 Tile，再按与 Auto 相同的接口调用计算/访存指令。
