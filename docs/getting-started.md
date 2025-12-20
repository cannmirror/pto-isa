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

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 python3-pip python3-venv git
```

### Linux (RHEL/CentOS/Rocky)

```bash
sudo dnf groupinstall -y "Development Tools" || true
sudo dnf install -y cmake ninja-build python3 python3-pip git
```

### Windows

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

## Python Environment (recommended)

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

## Run CPU Simulator (recommended first step)

This builds and runs the CPU ST test binaries under `tests/cpu/st` and executes all testcases:

```bash
python3 run_cpu.py --clean --verbose
```

Common options:

- Run a single testcase:

  ```bash
  python3 run_cpu.py --testcase tadd
  ```

- Run a single gtest case:

  ```bash
  python3 run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
  ```

- Build & run the GEMM demo:

  ```bash
  python3 run_cpu.py --demo gemm --verbose
  ```

- Build & run the Flash Attention demo:

  ```bash
  python3 run_cpu.py --demo flash_attn --verbose
  ```

## (Optional) Ascend CANN Environment (Linux)

If you plan to run NPU or simulator STs, install Ascend drivers + CANN toolkit (see Ascend docs for your distribution), then source `setenv.bash`:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

Then use the repo scripts, for example:

```bash
chmod +x run_st.sh
./run_st.sh a5 npu simple
```

## Next steps

- ISA overview: `docs/PTOISA.md`
- Instruction reference: `docs/isa/README.md`
- PTO assembly syntax (PTO-AS): `docs/grammar/PTO-AS.md`
