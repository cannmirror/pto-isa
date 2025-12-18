# Matrix Multiply

This page groups the matmul family of instructions.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Shared notation and events: `docs/isa/conventions.md`

| Instruction | Summary | Reference |
| :-- | :-- | :-- |
| `TMATMUL` | GEMM producing an accumulation tile | `docs/isa/TMATMUL.md` |
| `TMATMUL_ACC` | GEMM accumulating into an existing accumulation tile | `docs/isa/TMATMUL_ACC.md` |
| `TMATMUL_BIAS` | GEMM with a bias tile | `docs/isa/TMATMUL_BIAS.md` |
