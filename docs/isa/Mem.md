# 访存操作（GM ↔ Tile）
GM（全局内存）与 Tile 间的数据加载/存储。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TLOAD | GM → Tile 加载 | `TLOAD %Global -> %Tile` | docs/isa/TLOAD.md |
| TSTORE | Tile → GM 存储（可原子/量化） | `TSTORE %Tile -> %Global` | docs/isa/TSTORE.md |

约束要点：GM 布局/对齐需符合要求；Tile 形状与数据类型需匹配；原子/预量化参数按实现限制使用。
