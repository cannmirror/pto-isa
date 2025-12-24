<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# Getting Started

This guide covers prerequisites and setup on **macOS / Linux / Windows**, and shows how to build and run the **CPU simulator** first (recommended). Running on Ascend (NPU / simulator) requires Ascend CANN and is typically **Linux-only**.

## Prerequisites

### Required (CPU simulator)

- Git
- Python `>= 3.8` (3.10+ recommended)
- CMake `>= 3.16`
- A C++ compiler with C++23 support:
  - Linux: GCC 11+ or Clang 16+
  - macOS: Xcode/AppleClang (or Homebrew LLVM)
  - Windows: Visual Studio 2022 Build Tools (MSVC)
- Python packages: `numpy` (the CPU test data generators use it)

`run_cpu.py` can install `numpy` automatically (unless you pass `--no-install`).

### Optional (faster builds)

- Ninja (CMake generator)
- A working internet connection (CMake may fetch GoogleTest for CPU ST tests if not installed system-wide)

## OS Setup

### macOS

- Install Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

- Install dependencies (recommended via Homebrew):

  ```bash
  brew install cmake ninja python
  ```

If you do not use Homebrew, make sure `python3`, `cmake`, and a modern `clang++` are on `PATH`.

### Linux (Ubuntu 20.04)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```
### Windows(TODO)

Install the following:

- Git for Windows
- Python 3 (and ensure it’s on `PATH`)
- CMake
- Visual Studio 2022 Build Tools (Desktop development with C++)

Using `winget` (optional):

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

After installation, open a **Developer Command Prompt for VS 2022** (or ensure `cl.exe` is on `PATH`).

## Get The Code

```bash
git clone <YOUR_REPO_URL>
cd pto-tile-lib
```

## Python Environment

Create and activate a virtual environment:

- macOS / Linux:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  python -m pip install numpy
  ```

- Windows (PowerShell):

  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install numpy
  ```

## Run CPU Simulator

This builds and runs the CPU ST test binaries under `tests/cpu/st` and executes all testcases:

```bash
python3 tests/run_cpu.py --clean --verbose
```

Common options:

- Run a single testcase:

  ```bash
  python3 tests/run_cpu.py --testcase tadd
  ```

- Run a single gtest case:

  ```bash
  python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

- Build & run the GEMM demo:

  ```bash
  python3 tests/run_cpu.py --demo gemm --verbose
  ```

- Build & run the Flash Attention demo:

  ```bash
  python3 tests/run_cpu.py --demo flash_attn --verbose
  ```

- Optional:

  ```bash
  # specify the cxx path
  python3 tests/run_cpu.py --cxx=/path/to/compiler
  ```

  ```bash
  # print detail logs
  python3 tests/run_cpu.py --verbose
  ```

  ```bash
  # clean up the build directory
  python3 tests/run_cpu.py --clean
  ```

  ```bash
  # on Windows, maybe need specify generator and cmake_perfix_path
  python3 tests/run_cpu.py --clean --generator "MinGW Makefiles" --cmake_prefix_path D:\gtest\
  ```

# Environment Setup (Ascend 910B/910C, Linux)

## Prerequisites

Before using this project, make sure the following basic dependencies and the NPU driver/firmware are installed.

1. **Install build dependencies**

   The project requires the following dependencies for building from source (please pay attention to the version requirements):

   - Python >= 3.8.0
   - GCC >= 7.3.0
   - CMake >= 3.16.0
   - GoogleTest (only required when running unit tests; recommended version:
     [release-1.14.0](https://github.com/google/googletest/releases/tag/v1.14.0))

        After downloading the
        [GoogleTest source](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz),
        install it with:

        ```bash
        tar -xf googletest-1.14.0.tar.gz
        cd googletest-1.14.0
        mkdir temp && cd temp                # create a temp build dir under the googletest source tree
        cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
        make
        make install                         # install as root
        # sudo make install                  # install as a non-root user
        ```
      > **Note**
      > 
      > Python needs to download packages such as os, numpy, ctypes, struct, copy, math, enum, ml_dtypes, en_dtypes, etc.
      > 
      > If you have already installed googletest by other means, you need to make the corresponding changes to the CMakeLists.txt. For examle, you used `cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"` when installing googletest, you need to add `add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)` in tests/npu/[a2a3 | a5]/src/st/CMakeLists.txt

2. **Install driver and firmware (runtime dependency)**

   The driver and firmware are required to run operators. If you only need to build, you can skip this step.
   For installation guidance, see:
   [NPU Driver and Firmware Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=VmIns&OS=Ubuntu&Software=cannToolKit).

## Install Software Packages

This project supports building from source. Before building, prepare the environment as follows.

1. **Install the community edition CANN toolkit**

    Download the appropriate `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` installer for your environment.
    
    ```bash
    # Ensure the installer is executable
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # Install
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`: the CANN toolkit version.
    - `${arch}`: the CPU architecture, such as `aarch64` or `x86_64`.
    - `${install_path}`: the installation path.
    - If `--install-path` is omitted, the default path is used. If installed as root, the software is placed under
      `/usr/local/Ascend/latest`. If installed as a non-root user, it is placed under `$HOME/Ascend/latest`.


## Environment Variables

- Default path (installed as root)

    ```bash
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```

- Default path (installed as a non-root user)
    ```bash
    source $HOME/Ascend/latest/bin/setenv.bash
    ```

- Custom installation path
    ```bash
    source ${install_path}/latest/bin/setenv.bash
    ```

## Source Code Download

Download the source with:
```bash
# Clone the repository (master branch as an example)
git clone https://gitcode.com/cann/pto-tile-lib
```
