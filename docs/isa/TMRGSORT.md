# TMRGSORT

## 说明
**归并排序（Merge Sort）**

- 功能：归并排序（多路/块）

对多个已排序片段执行归并排序，重载形式支持 2/3/4 路归并及块级归并。

---

## 汇编语法
```asm
TMRGSORT %Src... , -> %Dst
```

### 汇编符号说明
- `%SrcTile/%SrcTile0/%SrcTile1`：输入 Tile（数量与指令匹配）。
- `%DstTile`：输出 Tile。
- `%R`：标量立即数/寄存器（仅标量类指令使用）。
- `cmpMode/rmode/selectMode`：模式修饰或参数（具体含义见 C++ 接口与实现约束）。

---

## C++ Intrinsic 接口
```cpp
template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, typename Src3TileData,
          bool exhausted>
PTO_INST void TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1,
         Src2TileData &src2, Src3TileData &src3);

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, bool exhausted>
PTO_INST void TMRGSORT(DstTileData &dst,
                            MrgSortExecutedNumList &executedNumList,
                            TmpTileData &tmp, Src0TileData &src0,
                            Src1TileData &src1, Src2TileData &src2);

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, bool exhausted>
PTO_INST void TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1);

template <typename DstTileData, typename SrcTileData>
PTO_INST void TMRGSORT(DstTileData &dst, SrcTileData &src,
                            uint32_t blockLen);
```

### 参数说明
| 参数 | 含义 |
| ------ | ----------------------------------------- |
| dst | 输出 Tile（写入结果） |
| executedNumList | 归并执行计数列表（实现定义） |
| tmp | 临时 Tile（用于中间结果） |
| src0 | 输入 Tile 0 |
| src1 | 输入 Tile 1 |
| src | 输入 Tile |
| blockLen | 块长（需满足实现约束） |

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
- 支持数据类型（编译期检查）：float
- 运行期约束：
  - ERROR: Total memory usage exceeds UB limit!

### 实现检查（A5）
- 支持数据类型（编译期检查）：float

---

## 编程示例
### PTO Auto 写法
```cpp
#include "pto/common/pto_instr.hpp"
#include "pto/common/pto_tile.hpp"

using namespace pto;
// 归并排序的重载较多，以下仅示意 blockLen 版本
template <typename T>
void example() {
  using Data = Tile<TileType::Vec, T, 16, 16, BLayout::RowMajor>;
  Data src, dst;
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

### PTO Manual 写法（可选）
- 若启用手动模式并需要显式分配片上地址，可先使用 `TASSIGN` 绑定 Tile，再按与 Auto 相同的接口调用计算/访存指令。
