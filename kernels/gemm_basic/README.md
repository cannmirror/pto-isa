# 基础GEMM算子样例

## 概述

本样例以gemm样例为例，介绍如何使用PTO快速开发实现gemm算子，帮助开发者理解和掌握使用PTO开发算子的搭建、编译、运行等过程。

## 支持的AI处理器

- Ascend 910C
- Ascend 910B

## 目录结构介绍

```
├── gemm_basic
│   └── scripts
│       └── gen_data.py                    // 输入数据和真值数据生成脚本文件
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── gemm_basic_kernel.cpp              // 算子kernel实现
│   ├── main.cpp                           // 主函数，调用算子的应用程序
│   └── run.sh                             // 执行脚本
```

## 算子描述

### 算子功能

  本样例中实现的是[m, k, n]固定为[512, 2048, 1536]的gemm算子。
  gemm算子的数学表达式为：
  $$
  C = A * B
  $$
  其中A的形状为[512, 2048], B的形状为[2048, 1536], C的形状为[512, 1536]。
  
### 算子规格

  在本样例中，算子实现支持的shape为：m = 512, k = 2048, n = 1536。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">gemm</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">m * k</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">n * k</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">m * n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gemm_basic_kernel</td></tr>
  </table>

### Tiling参数

本样例执行平台有24核，因此考虑将数据均匀切分到24核上执行，以实现并行计算，提高计算效率。切分过程本着分核优先切m、n的原则，考虑将m切为4份，n切为6份，用满24核。切分之后单核的shape为：singleCoreM = 128, singleCoreK = 2048, singleCoreN = 256。在每个核中，上述单核shape依然超出L0内存大小，因此需在单核中再次切分，考虑到搬出时GM地址512Byte对齐，这里按基本块大小为64将k切分，切分之后的基本块shape为：baseM = 128, baseK = 64, baseN = 256。

| 参数名       | Value     |
| ---------- | ------------ |
| m | 512 |
| k | 2048 |
| n | 1536 |
| singleCoreM | 128 |
| singleCoreK | 2048 |
| singleCoreN | 256 |
| baseM | 128 |
| baseK | 64 |
| baseN | 256 |

## 算子实现

### 数据类型定义

在算子的实现中，需要首先定义矩阵在gm、L1、L0上的数据类型，并为其分配相应的存储空间。这些定义和分配是实现高效矩阵运算的基础，确保数据在不同阶段能够正确处理。具体定义如下：

```cpp
  using NDValidShapeA = TileShape2D<U, baseM, baseK>;
  using NDsingleCoreShapeA = BaseShape2D<U, M, K>;
  using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA>; //定义gm上A矩阵，默认layout为ND

  using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
  using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
  using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>; //定义gm上B矩阵

  using NDValidShapeC = TileShape2D<T, baseM, baseN>;
  using NDWholeShapeC = BaseShape2D<T, M, N>;
  using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC>; //定义gm上输出矩阵C

  using TileMatAData = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>; //定义L1A矩阵的Tile结构，大分型列优先，小分型行优先
  using TileMatBData = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>; //定义L1B矩阵的Tile结构，大分型行优先，小分型列优先

  TileMatAData aMatTile[2]; // L1A double buffer
  TileMatBData bMatTile[2]; // L1B double buffer
  TASSIGN(aMatTile[0], 0x0); // 为L1上的4块buffer分配内存空间: aMatTile[0], aMatTile[1], bMatTile[0], bMatTile[1] 
  TASSIGN(aMatTile[1], 0x0 + baseM * baseK  * sizeof(U));
  TASSIGN(bMatTile[0], 0x0 + baseM * baseK * 2 * sizeof(U));
  TASSIGN(bMatTile[1], 0x0 + baseM * baseK * 2 * sizeof(U) + baseK * baseN * sizeof(U));

  using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>; //左矩阵L0A
  using RightTile = TileRight<S, baseK, baseN, baseK, baseN>; //右矩阵L0B
  using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>; //输出矩阵L0C

  LeftTile aTile[2]; // L0A double buffer
  RightTile bTile[2]; // L0B double buffer
  ResTile cTile;
  TASSIGN(aTile[0], 0x0); // 把L0A上的内存空间分配给aTile[0]和aTile[1]
  TASSIGN(aTile[1], 0x0 + baseM * baseK  * sizeof(U));
  TASSIGN(bTile[0], 0x0); // 把L0B上的内存空间分配给bTile[0]和bTile[1]
  TASSIGN(bTile[1], 0x0 + baseK * baseN  * sizeof(U));
  TASSIGN(cTile, 0x0); // L0C分配空间
```

### 流水排布

本样例主要涉及的流水包括数据搬入流水、cube计算和数据搬出流水。为了优化硬件资源利用率和计算效率，样例中在L1和L0缓存中采用了double buffer机制，使得数据搬运与cube计算流水并行，提升cube利用率和计算效率。流水排布的核心在于指令的执行顺序，需要通过手动插入同步指令来确保各流水环节之间的正确性和高效协作。具体来说，本样例涉及的同步包括:

- 数据同步:MTE2->MTE1同步，MTE1->mmad同步，mmad->fixpipe同步  
- 反向同步：MTE1->MTE2同步，mmad->MTE1同步  

流水示意图如下：  
![gemm流水示意图](../../docs/figures/gemm_pipeline.png)

## 编译运行

- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。

  ```bash
  cd ${git_clone_path}/demos/baseline/gemm_basic
  ```

  请根据当前环境上CANN开发套件包，选择对应配置环境变量的命令。执行以下命令统一配置环境变量。

  ```bash
  # 配置CANN环境变量
  source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
  ```

- 样例执行  

  ```bash
  bash run.sh -r npu -v Ascend910B1
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test success
  ```
