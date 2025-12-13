# TLOAD

## 说明
**加载（Load）**

- 功能：GM → Tile 加载

从 `GlobalTensor` 读取并填充到 `dst` Tile（超出有效范围时的填充值由 Tile/实现定义）。

---

## 汇编语法
```asm
TLOAD %Global , -> %Tile
```

### 汇编符号说明
- `%SrcTile/%SrcTile0/%SrcTile1`：输入 Tile（数量与指令匹配）。
- `%DstTile`：输出 Tile。
- `%R`：标量立即数/寄存器（仅标量类指令使用）。
- `cmpMode/rmode/selectMode`：模式修饰或参数（具体含义见 C++ 接口与实现约束）。

---

## C++ Intrinsic 接口
```cpp
template <typename TileData, typename GlobalData>
PTO_INST void TLOAD(TileData &dst, GlobalData &src);
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
- 编译期约束：
  - TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ!
  - TLOAD(MatTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ/ND2NZ/DN2ZN!
  - GlobalTensor ony support 2 dim when ND2NZ!
  - TileData ony support SFractalSize = 512Bytes!
  - GlobalTensor ony support 2 dim when DN2ZN!
  - TileData ony support SFractalSize = 512Bytes!
- 运行期约束：
  - The shape of src and dst must be greater than 0!

### 实现检查（A5）
- 支持数据类型（编译期检查）：int64_t, uint64_t
- 编译期约束：
  - Data type must be b8/b16/b32/b64
  - TileData::PadVal only support Null or Zero in B64 mode
  - Source dtype must be same with dst dtype!
  - Src and dst layout must be same!
  - Src GlobalTensor Col and Tile ValidCol must be the same!
  - Src GlobalTensor Row Products and Tile ValidRow must be the same!

---

## 编程示例
### PTO Auto 写法
```cpp
#include "pto/common/pto_instr.hpp"
#include "pto/common/pto_tile.hpp"

using namespace pto;
template <typename T>
void example(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16, BLayout::RowMajor>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;
  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### PTO Manual 写法（可选）
- 若启用手动模式并需要显式分配片上地址，可先使用 `TASSIGN` 绑定 Tile，再按与 Auto 相同的接口调用计算/访存指令。
