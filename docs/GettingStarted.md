# 环境部署

## 前提条件

使用本项目前，请确保如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.8.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - googletest（仅执行UT时依赖，建议版本 [release-1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0)）

        下载[googletest源码](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz)后，执行以下命令安装：

        ```bash
        tar -xf googletest-1.14.0.tar.gz
        cd googletest-1.14.0
        mkdir temp && cd temp                # 在googletest源码根目录下创建临时目录并进入
        cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
        make
        make install                         # root用户安装googletest
        # sudo make install                  # 非root用户安装googletest
        ```

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导详见《[安装NPU驱动和固件指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=VmIns&OS=Ubuntu&Software=cannToolKit)》。

## 软件包安装

本项目支持由源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **安装社区版CANN toolkit包**

    根据实际环境架构，获取对应的`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径。
    - 缺省--install-path时， 则使用默认路径安装。若使用root用户安装，安装完成后相关软件存储在“/usr/local/Ascend/latest”路径下；若使用非root用户安装，安装完成后相关软件存储在“$HOME/Ascend/latest”路径下。


## 环境变量配置

- 默认路径，root用户安装

    ```bash
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```

- 默认路径，非root用户安装
    ```bash
    source $HOME/Ascend/latest/bin/setenv.bash
    ```

- 指定路径安装
    ```bash
    source ${install_path}/latest/bin/setenv.bash
    ```

## 源码下载

开发者可通过如下命令下载本仓源码：
```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/pto-tile-lib
```