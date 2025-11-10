# pto-tile-lib

## 快速上手
## 编译运行st用例
 
  - 配置环境变量
 
    以CANN社区包为例。社区包安装路径：/usr/local/Ascend/ascend-toolkit/latest
    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
    依赖gtest，下载gtest[源码](https://github.com/google/googletest.git)后执行以下命令安装，建议安装1.14以上的版本：
    ```
    mkdir temp && cd temp                 # 在gtest源码根目录下创建临时目录并进入
    cmake ..
    make
    make install                         # root用户安装gtest
    # sudo make install                  # 非root用户安装gtest
    ```

  - 编译执行
 
    ```
    python3 test/script/run_st.py -r [RUN_MODE] -v [SOC_VERSION] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
    ```
    其中脚本参数说明如下：
    - RUN_MODE ：编译执行方式，NPU仿真，NPU上板，对应参数分别为[sim / npu]。
    - SOC_VERSION ：昇腾AI处理器平台，目前仅支持传入[a3 / a5]。
    - TEST_CASE ：对应要执行的用例。
    - GTEST_FILTER_CASE ：可选参数，需要执行用例中的某个具体用例。

    示例如下:
    A3:
    ```
    python3 test/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
    ```
    A5:
    ```
    python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
    ```
