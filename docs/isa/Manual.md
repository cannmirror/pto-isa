# 手动模式 / 资源绑定（Manual Op）
当前 `include/pto/common/pto_instr.hpp` 仅导出 `TASSIGN`，用于显式绑定 Tile 缓冲地址。若后续开放 TSYNC/事件模型，请在此补充。

| 指令 | 功能 | 接口示例 | 单指令文档 |
| :-- | :-- | :-- | :-- |
| TASSIGN | 显式绑定 Tile 缓冲地址 | `TASSIGN %Tile, %addr` | docs/isa/TASSIGN.md |

注意：与编译器自动分配的缓冲区需避免冲突；地址参数需满足对齐和权限要求。
