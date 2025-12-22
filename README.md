<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Lib

Parallel Tile Operation (PTO) is a virtual instruction set architecture designed by Ascend CANN, focusing on Tile-level operations. This repository offers high-performance, cross-platform Tile operations based on different Ascend platforms. By porting to PTO instruction sequences, users can migrate Ascend hardware easily.

## News

* **2025-12-31**: PTO Tile Lib becomes publicly available.

## Overview

The PTO ISA (Instruction Set Architecture) is built on Ascend’s underlying hardware and software abstraction, providing over 90 standard Tile-level operations. 

Ascend hardware architectures have significantly evolved over generations, leading to major changes in the instruction sets. The PTO instruction set bridges these hardware differences by raising the abstraction level. We ensure that these PTO instructions work correctly across platforms while maintaining backward compatibility. However, this abstraction does not hide performance tuning opportunities. Users can still fine-tune performance by adjusting Tile sizes, Tile shapes, instruction order, Tile transformations, and other Tile operations. This provides sufficient control to fine-tune internal pipeline flows.

Our goal is to offer users a simplified, yet powerful way to optimize performance, enabling them to write high-performance code with PTO instructions.

Currently, PTO instructions are integrated into the following frameworks:

* [PyPTO](https://gitcode.com/cann/pypto/)
* [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/)

## Target Users of this Repository

PTO Tile Library is not aimed at beginner-level users. The intended audience includes:

* Backend developers implementing frameworks that directly interface with Ascend hardware.
* Cross-platform application developers (NPU & GPU).
* High-performance operator developers (manual operator implementations).

## Design Philosophy of PTO Instructions

* **Tile as the Standard Unit**: Tile is a fixed unit that cannot be indexed by pointers, making it easier for compilers to perform alias analysis.
* **Static to Dynamic Conversion, Code Specialization**: Tiles are implemented without address or index calculations, removing dynamic Masks.
* **Reduced Flexibility for Higher Performance**: We write high-performance operators for standard Tile shapes, using the "container theory" where multiple choices for achieving the same performance are reduced. This minimizes the complexity for compilers and users.

## How to Use PTO Tile Library

PTO instructions support two modes: **Auto Mode** (where the user does not allocate buffers or manage pipelining) and **Manual Mode** (where the user must allocate buffer addresses and manage pipelining). We recommend the following steps for optimizing operators:

1. Develop the operator based on Auto Mode, generating PTO instruction sequences according to the algorithm logic.
2. Verify functionality and correctness in CPU simulations.
3. Port the code to Ascend hardware to ensure correctness and collect performance data.
4. Identify performance bottlenecks (CUBE Bound, MTE Bound, Vector Bound) and begin optimization and tuning.

We ensure that each PTO instruction, when implemented within a fixed Tile shape, fully leverages the capabilities of the underlying hardware. We encapsulate low-level hardware implementations into the Tile abstractions and utilize expert knowledge to create a variety of Tile templates. During static compilation, the compiler selects the best assembly implementation for the current shape based on template parameters. By merging different PTO instructions, we achieve optimal performance.

In this repository, we demonstrate how standard Tile operations can be mapped to various pipelines through template parameters:

* Static Tile Shape (Row, Col)
* Dynamic Tile Mask (Valid Mask)
* Event Record & Wait (Set wait flag)
* Specialized Fixed Function (SFU)
* Fixed Pipeline (FIXP)

PTO ISA defines over 90 standard operations. This repository currently covers and implements 46 different instructions, with ongoing efforts to add more.

## Platform Support

* Ascend (A3 / A5)
* CPU (X86 / AArch64)

## Quickstart Guide

For detailed, OS-specific setup (Windows / Linux / macOS), see: [docs/getting-started.md](docs/getting-started.md).

### Run CPU Simulator (recommended first step)

CPU simulation is cross-platform and does not require Ascend drivers/CANN:

```bash
python3 run_cpu.py --clean --verbose
```

Build & run the GEMM demo (optional):

```bash
python3 run_cpu.py --demo gemm --verbose
```

Build & run the Flash Attention demo (optional):

```bash
python3 run_cpu.py --demo flash_attn --verbose
```

### Running a Single ST Test Case

```bash
python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]
```

Example:

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

### Running Recommended Test Suites

```bash
# Execute the following commands from the project root directory:
chmod +x ./tests/run_st.sh
./tests/run_st.sh a5 npu simple
./tests/run_st.sh a3 sim all
```

### Running CPU Simulation Tests

```bash
# Execute the following commands from the project root directory:
chmod +x ./tests/run_cpu_tests.sh
./tests/run_cpu_tests.sh

python3 ./tests/run_cpu.py --verbose
```

## Build / Run Instructions (Reference Repository Scripts)

### Configuring Environment Variables (Ascend CANN)

For example, if using the CANN community package and installing to `/usr/local/Ascend/ascend-toolkit/latest`:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

### One-click Build and Run (Optional)

* Run Full ST Tests:

  ```bash
  chmod +x build.sh
  ./build.sh --run_all --a3 --sim
  ```
* Run Simplified ST Tests:

  ```bash
  chmod +x build.sh
  ./build.sh --run_simple --a5 --npu
  ```
* Packaging:

  ```bash
  chmod +x build.sh
  ./build.sh --pkg
  ```

## Documentation

* ISA Guide and Instruction Navigation: [docs/README.md](docs/README.md)
* ISA Instruction Documentation Index: [docs/isa/README.md](docs/isa/README.md)
* Developer Coding Documentation Index: [docs/coding/README.md](docs/coding/README.md)
* Getting Started Guide (recommended to run on CPU before moving to NPU): [docs/getting-started.md](docs/getting-started.md)
* Security and Disclosure Process: [SECURITY.md](SECURITY.md)
* Directory-level Reading (Code Organization):

  * Build and Packaging (CMake): [cmake/README.md](cmake/README.md)
  * External Header Files and APIs: [include/README.md](include/README.md), [include/pto/README.md](include/pto/README.md)
  * NPU Implementation (Split by SoC): [include/pto/npu/README.md](include/pto/npu/README.md), [include/pto/npu/a2a3/README.md](include/pto/npu/a2a3/README.md), [include/pto/npu/a5/README.md](include/pto/npu/a5/README.md)
  * Kernel/Custom Operators: [kernels/README.md](kernels/README.md), [kernels/custom/README.md](kernels/custom/README.md)
  * Testing and Use Cases: [tests/README.md](tests/README.md), [tests/script/README.md](tests/script/README.md)
  * Packaging Scripts: [scripts/README.md](scripts/README.md), [scripts/package/README.md](scripts/package/README.md)

## Repository Structure

* `include/`: PTO C++ header files (see [include/README.md](include/README.md))
* `kernels/`: Custom operators and kernel implementations (see [kernels/README.md](kernels/README.md))
* `docs/`: ISA instructions, API guidelines, and examples (see [docs/README.md](docs/README.md))
* `tests/`: ST/CPU test scripts and use cases (see [tests/README.md](tests/README.md))
* `scripts/`: Packaging and release scripts (see [scripts/README.md](scripts/README.md))
* `build.sh`, `run_st.sh`: Build, package, and example run entry points

## License

This project is licensed under the CANN Open Software License Agreement Version 2.0. See the `LICENSE` file for details.
