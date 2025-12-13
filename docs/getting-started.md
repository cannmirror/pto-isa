# 入门指南：先在 CPU 跑通，再跑 NPU

本文给出最小可行的环境与命令，先在本地 CPU 仿真（不依赖 Ascend 设备），再在 NPU（或 NPU 仿真器）跑通 ST 用例。

## 1. 环境准备
- 工具链：CMake ≥ 3.16、Python ≥ 3.8、clang++ 或 g++（建议 clang++）、git
- 单元测试：gtest ≥ 1.14（可 `apt/yum install libgtest-dev` 或源码编译 `make && make install`）
- Ascend 相关（仅 NPU / 模拟器阶段需要）：
  - CANN Toolkit：安装在 `/usr/local/Ascend/ascend-toolkit/latest`（或自定义路径）
  - 设置环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
  - 环境变量：`ASCEND_HOME_PATH` 指向上述路径

## 2. 仓库结构速览
- `tests/cpu/st`：CPU 侧 ST 用例（cmake + gtest）
- `tests/npu/a2a3/src/st`、`tests/npu/a5/src/st`：NPU 侧 ST 用例
- `tests/script/run_st.py`：NPU/模拟器一键构建与执行脚本
- `include/pto`：PTO ISA 对外头文件
- `cmake/third_party`：第三方依赖下载脚本

## 3. 在 CPU 跑通 ST 用例
1) 安装 gtest（若未安装）  
   ```bash
   mkdir /tmp/gtest-build && cd /tmp/gtest-build
   cmake /path/to/googletest -DBUILD_SHARED_LIBS=ON
   make -j$(nproc) && sudo make install
   ```
2) 构建与运行（示例使用 clang++，替换为你的编译器路径）  
   ```bash
   cd tests/cpu/st
   mkdir -p build && cd build
   cmake -DCMAKE_COMPILER=$(which clang++) -DCMAKE_BUILD_TYPE=Release ..
   make -j$(nproc)
   # 可直接运行生成的可执行文件（位于 build/bin/）
   ls bin/
   ./bin/<cpu_st_binary> --gtest_output=xml:./report.xml
   ```
   如 gtest 未安装在系统默认路径，补充 `-DCMAKE_PREFIX_PATH=/usr/local`（或实际安装前缀）。

## 4. 在 NPU 仿真器（sim）跑通 ST 用例
1) 先确保已 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并设置 `ASCEND_HOME_PATH`。  
2) 执行脚本（以 A3 模拟器、tmatmul 用例为例）：  
   ```bash
   python3 tests/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case1
   ```
   说明：  
   - `-r sim`：使用模拟器运行；脚本会自动调整 `LD_LIBRARY_PATH` 并加载 simulator 库。  
   - `-v a3`/`-v a5`：芯片版本。  
   - `-t`：用例目录名（位于对应的 `tests/npu/<chip>/src/st/testcase/` 下）。  
   - `-g`：gtest 过滤（可选，省略则跑该用例下全部 case）。  

## 5. 在 NPU 实机（npu）跑通 ST 用例
1) Ascend 驱动与 CANN 已安装，`set_env.sh` 已生效。  
2) 执行脚本（以 A3 实机、tmatmul 用例为例）：  
   ```bash
   python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
   ```
   说明：  
   - `-r npu`：在实机执行；无需模拟器 stub。  
   - 其余参数与 sim 相同。  

## 6. 常见问题
- gtest 未找到：确认已安装并在 CMake 路径可见，必要时增加 `-DCMAKE_PREFIX_PATH`。  
- 模拟器库未找到：确认 `ASCEND_HOME_PATH` 指向 CANN 安装目录，且已执行 `set_env.sh`。  
- 构建缓存问题：脚本会清理 `build/`，如需手动清理请删除对应 `build` 目录后重试。  

如需新增用例，推荐参考对应芯片目录下现有 `testcase` 结构与 CMake 配置。
