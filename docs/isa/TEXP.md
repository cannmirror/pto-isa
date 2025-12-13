# TEXP

## 说明
**指数（Tile Exp）**

- 功能：指数

$$ \mathrm{dst}_{i,j} = e^{\mathrm{src}_{i,j}} $$

---

## 汇编语法
```asm
TEXP %Src , -> %Dst
```

### 汇编符号说明
- `%SrcTile/%SrcTile0/%SrcTile1`：输入 Tile（数量与指令匹配）。
- `%DstTile`：输出 Tile。
- `%R`：标量立即数/寄存器（仅标量类指令使用）。
- `cmpMode/rmode/selectMode`：模式修饰或参数（具体含义见 C++ 接口与实现约束）。

---

## C++ Intrinsic 接口
```cpp
template <typename TileData>
PTO_INST void TEXP(TileData &dst, TileData &src);
```

### 参数说明
| 参数 | 含义 |
| ------ | ----------------------------------------- |
| dst | 输出 Tile（写入结果） |
| src | 输入 Tile |

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
- 支持数据类型（编译期检查）：float32_t, float, half, float16_t
- 编译期约束：
  - TEXP: Invalid data type
  - TEXP: TileType of src and dst tiles must be TileType::Vec.
  - TEXP: Number of valid columns must not be greater than number of tile columns.
  - TEXP: Number of valid rows must not be greater than number of tile rows.
- 运行期约束：
  - TEXP: Number of columns of src and dst must be the same.
  - TEXP: Number of rows of src and dst must be the same.

### 实现检查（A5）
- 编译期约束：
  - TEXP: not supported Layout type
  - TEXP: not supported Layout type

---

## 编程示例
### PTO Auto 写法
```cpp
#include "pto/common/pto_instr.hpp"
#include "pto/common/pto_tile.hpp"

using namespace pto;
// 示例：Vec Tile 上的逐元素/一元操作
template <typename T>
void example(__gm__ T* out, __gm__ T* in0, __gm__ T* in1) {
  using TileT = Tile<TileType::Vec, T, 16, 16, BLayout::RowMajor>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor g0(in0);
  GTensor g1(in1);
  GTensor gout(out);
  TileT t0, t1, td;

  TLOAD(t0, g0);
  TLOAD(t1, g1);

  // 在此处替换为目标指令，例如：
  // TADD(td, t0, t1);

  TSTORE(gout, td);
}
```

### PTO Manual 写法（可选）
- 若启用手动模式并需要显式分配片上地址，可先使用 `TASSIGN` 绑定 Tile，再按与 Auto 相同的接口调用计算/访存指令。
