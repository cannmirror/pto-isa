# Getting Started: CPU First, Then NPU

This guide describes a minimal workflow to validate PTO Tile Lib on CPU (simulation) first, then run NPU ST (simulator or on-board).

## Prerequisites

- CMake >= 3.16
- A C++ compiler (clang++ or g++)
- Python >= 3.8
- Git

For NPU (simulator / on-board), you also need the Ascend CANN Toolkit installed and sourced (example path may differ):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Repository landmarks

- CPU ST: `tests/cpu/st/`
- NPU ST: `tests/npu/a2a3/src/st/`, `tests/npu/a5/src/st/`
- Test scripts: `tests/script/` (especially `tests/script/run_st.py`)
- Public headers: `include/pto/`
- ISA docs: `docs/isa/`

## Run CPU ST (recommended first step)

Build and run the CPU simulation ST suite:

```bash
cd tests/cpu/st
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
ctest --output-on-failure
```

If you prefer running all CPU suites via the helper script, see `tests/script/all_cpu_tests.py`.

## Run NPU ST (simulator or on-board)

Use the script entrypoint:

```bash
python3 tests/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case1
```

Key flags:

- `-r sim|npu`: run on simulator or on-board device
- `-v a3|a5`: target SoC variant
- `-t <testcase>`: testcase directory under `tests/npu/<chip>/src/st/testcase/`
- `-g <gtest_filter>`: optional gtest filter

For more examples, see `tests/script/README.md`.
