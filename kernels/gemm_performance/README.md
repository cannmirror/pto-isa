# 高性能GEMM算子样例
## 概述
本样例介绍如何使用PTO编写高性能GEMM算子。
## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── gemm_performance
│   └── scripts
│       └── gen_data.py                 // 输入数据和真值数据生成脚本文件
│   ├── CMakeLists.txt                  // 编译工程文件
│   ├── gemm_performance_kernel.cpp  // 算子kernel实现
│   └── main.cpp                        // 主函数，调用算子的应用程序
```
## 算子描述
### 算子功能：  

  本样例中实现的是[m, k, n]固定为[6144, 6144, 6144]的GEMM算子。 
  GEMM算子的数学表达式为：
  $$
  C = A * B
  $$
  其中A的形状为[6144, 6144], B的形状为[6144, 6144], C的形状为[6144, 6144]。
### 算子规格：  

  在本样例中，算子实现支持的shape为：m = 6144, k = 6144, n = 6144。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">GEMM</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">m * k</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">n * k</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">m * n</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">GEMMPerformance</td></tr>
  </table>

### 算子实现：  
  GEMM算子主要的功能实现是矩阵乘法，其中主要包含的流水为数据搬入、Cube计算流水和搬出流水。GEMM高性能算子实现过程中，主要的优化措施有优化分核逻辑、优化基本块、使能L1缓存、使能Double Buffer。本样例中，以24核的Ascend 910B为性能验证平台。
  - 分核逻辑：  
    开启尽可能多的Cube核使能并行计算，通过合理的将数据分配给不同的核来执行任务。本样例执行平台有24核，因此需要考虑将数据切分到24核上执行，以实现并行计算。考虑到M、N、K一样大，优先考虑单核不切K，将M、N切分到24核上执行，可以考虑4*6分组，M方向切4份，N方向切6份，单核计算的M、K、N即singleCoreM、singleCoreK、singleCoreN分别为1536，6144，1024。或者考虑6*4分组，即M方向切6份，N方向切4份，singleCoreM、singleCoreK、singleCoreN则分别为1024，6144，1536。本样例选择第一种分组方式。 
  - 基本块选择：  
    选择最优的基本块参数，基本块参数为A、B、C矩阵中参与一次TMATMUL矩阵乘指令的shape大小，以元素为单位。基本块的选择原则为计算访存比最大，即在Cube计算量最大的情况下，访存的数据量最小。在输入为fp16类型的情况下，Cube执行单元1 cycle能算16 * 16 * 16个数。根据经验，基本块[baseM, baseN, baseK] = [128, 256, 64]和[128, 128, 128]两种切分方案均满足搬出时GM地址512Byte对齐（每搬出一次Matmul结果时，地址分别偏移256 * 4byte和128 * 4byte），Cube计算cycle数一致，为(128 * 64 * 256) / (16 * 16 * 16) = (128 * 128 * 128) / (16 * 16 * 16) = 512cycle。针对[baseM, baseN, baseK] = [128, 256, 64]，计算访存比为512cycle / (128 * 64 * 2 + 256 * 64 * 2) = 512cycle / 48KB；针对[baseM, baseN, baseK] = [128, 128, 128]，计算访存比为512cycle / (128 * 128 * 2 + 128 * 128 * 2) = 512cycle / 64KB；可见[128, 256, 64]基本块方案的计算访存比更高，计算密度更大，同样的计算量，需要的数据量最小，最大限度提高Cube单元的计算量。按照这个原则，这里我们选择基本块baseM，baseK，baseN为128，64，256，不选择128，128，128。
  - L1缓存：  
    GM到L1中的数据搬运过程中，一次搬入多个基本块到L1，称为L1缓存。通过合理控制搬运数据块的大小，提升带宽利用效率。本样例中我们选择的基本块大小baseM，baseK，baseN为128，64，256，结合L1大小并且考虑尽可能用满L1缓存，我们设置A和B矩阵都按K方向每次搬入4个数据块，即设置StepKa=StepKb=4，这样L1缓存中A矩阵会占用64KB， B矩阵会占用128K，加上L1中我们需要使能Double Buffer，最终L1缓存中A矩阵会占用128KB， B矩阵会占用256K。
  - Double Buffer： 
    搬运和计算过程中，通过使用双缓冲，使得数据搬运与Cube计算流水并行，减少Cube单元闲置，提升Cube利用率。本样例中，我们在L1、L0A和L0B上均使能Double Buffer。 
### Tiling参数
   按照如上分析的优化措施，确定如下的Tiling参数：
| 参数名       | Value     |
| ---------- | ------------ |
| m | 6144 |
| k | 6144 |
| n | 6144 |
| singleCoreM | 1536 |
| singleCoreK | 6144 |
| singleCoreN | 1024 |
| baseM | 128 |
| baseK | 64 |
| baseN | 256 |
| stepM | 1 |
| stepKa | 4 |
| stepKb | 4 |
| stepN | 1 |
### 实测性能
  在24核的Ascend 910B上，执行本GEMM算子样例，性能为： Cube利用率86%，MTE2利用率95%，执行耗时为： 1.504ms。


## 编译运行 
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/demos/baseline/gemm_performance
  ```
  请根据当前环境上CANN开发套件包，选择对应配置环境变量的命令。执行以下命令统一配置环境变量。
  ```bash
  # 配置CANN环境变量
  source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
  ```
- 输入及输出标杆构造 
  ```bash
  python3 scripts/gen_data.py
  ```
- 样例执行  
  ```bash
  bash run.sh -r npu -v Ascend910B1
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test success
  ```
  
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/12/15 | 样例目录调整，新增本readme |