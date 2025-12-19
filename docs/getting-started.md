<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# Getting Started

This guide walks you through installing prerequisites, setting up an Ascend CANN environment (optional), and building/running PTO Tile Lib. If you are new to the project, start with CPU simulation first.

## Prerequisites

### Build tools

- Python `>= 3.8`
- CMake `>= 3.16`
- A C++ compiler with C++23 support (recommended: `clang++` on macOS, `g++` on Linux)
- (Optional) GoogleTest `v1.14.0` (only required to build/run unit tests)

### Ascend runtime (only required for NPU/simulator runs)

- Ascend NPU driver + firmware
- Ascend CANN toolkit (Community Edition or commercial distribution)

## Install GoogleTest (optional)

If you need to run unit tests and GoogleTest is not available on your system, you can build and install it from source:

```bash
tar -xf googletest-1.14.0.tar.gz
cd googletest-1.14.0
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
cmake --build . -j
sudo cmake --install .
```

## Install Ascend CANN Toolkit (optional)

If you plan to run on Ascend hardware (or use the simulator), install the CANN toolkit package for your platform:

1. Download the package `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`.
2. Install it (example uses an explicit install path):

```bash
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

Where:

- `${cann_version}` is the CANN toolkit version.
- `${arch}` is the CPU architecture (e.g., `aarch64`, `x86_64`).
- `${install_path}` is the installation prefix you choose.

If `--install-path` is omitted, the installer uses the default location.

## Configure environment variables (Ascend CANN)

After installing CANN, source `setenv.bash` (path depends on your install location):

```bash
source /usr/local/Ascend/latest/bin/setenv.bash
```

Or, if installed under your home directory:

```bash
source "$HOME/Ascend/latest/bin/setenv.bash"
```

Or, if you installed to a custom `${install_path}`:

```bash
source "${install_path}/latest/bin/setenv.bash"
```

## Clone the repository

```bash
git clone https://gitcode.com/cann/pto-tile-lib
cd pto-tile-lib
```

## Run CPU simulation (recommended first step)

CPU simulation does not require Ascend drivers/CANN and is the fastest way to validate correctness locally:

```bash
python3 run_cpu.py --clean --no-install
```

## Next steps

- ISA overview: `docs/PTOISA.md`
- Instruction reference: `docs/isa/README.md`
- PTO assembly syntax (PTO-AS): `docs/grammar/PTO-AS.md`
