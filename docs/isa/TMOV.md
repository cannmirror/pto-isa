# TMOV

## 说明
**搬运/格式转换（Move/Convert）**

- 功能：Tile 间搬运/转换（可选模式/量化）

在 Tile 间搬运数据，可选模式/量化辅助参数（具体语义由重载/模板参数决定）。

---

## 汇编语法
```asm
TMOV %Src , -> %Dst
```

### 汇编符号说明
- `%SrcTile/%SrcTile0/%SrcTile1`：输入 Tile（数量与指令匹配）。
- `%DstTile`：输出 Tile。
- `%R`：标量立即数/寄存器（仅标量类指令使用）。
- `cmpMode/rmode/selectMode`：模式修饰或参数（具体含义见 C++ 接口与实现约束）。

---

## C++ Intrinsic 接口
```cpp
template <typename DstTileData, typename SrcTileData>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src);

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src);

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar);

template <typename DstTileData, typename SrcTileData, typename FpTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp);
```

### 参数说明
| 参数 | 含义 |
| ------ | ----------------------------------------- |
| dst | 输出 Tile（写入结果） |
| src | 输入 Tile |
| preQuantScalar | 预量化系数（实现定义） |
| fp | 浮点/量化辅助 Tile（实现定义） |

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
  - TMov: The shape of src needs to be the same as that of dst.
  - TMov: Invalid TileType.

### 实现检查（A5）
- 编译期约束：
  - TMov: The shape of destination and source tile must be the same.
  - TMov: DstTile Invalid Fractal.
  - TMov: DstTile Invalid Fractal.

---

## 编程示例
### PTO Auto 写法
```cpp
#include "pto/common/pto_instr.hpp"
#include "pto/common/pto_tile.hpp"

using namespace pto;
template <typename T>
void example() {
  using TileT = Tile<TileType::Vec, T, 16, 16, BLayout::RowMajor>;
  TileT src, dst;
  TMOV(dst, src);
}
```

### PTO Manual 写法（可选）
- 若启用手动模式并需要显式分配片上地址，可先使用 `TASSIGN` 绑定 Tile，再按与 Auto 相同的接口调用计算/访存指令。
