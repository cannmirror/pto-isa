# 8. Programming guide

## 8.1 Two programming styles

PTO kernels are commonly written in two styles:

### Auto style (productivity-first)

Source code expresses a dataflow:

- declare tiles
- `TLOAD → compute → TSTORE`

The compiler manages:

- tile storage assignment
- insertion of required synchronization

### Manual style (performance-first)

Source code explicitly manages:

- tile buffer assignment (`TASSIGN`)
- pipeline ordering (events/flags and/or `TSYNC`)
- double buffering and overlap

## 8.2 Execution model (SPMD vs MPMD)

PTO supports both SPMD and MPMD execution models:

- **SPMD**: all cores run the same entry function and use `block_idx` (and optionally `subblockid`) to select which
  region of the global tensor to process.
- **MPMD**: different cores (or groups of cores) may execute different tile programs, selected by the Device Machine
  scheduler (for example via a scheduler-provided `task_id`, or via separate entry points).

MPMD is typically used for multi-stage pipelines where each stage has different performance constraints (memory-bound
producer, compute-bound consumer, etc.). Each stage can still use SPMD tiling internally.

## 8.3 Recommended workflow

1. Write an Auto-style kernel and validate correctness on CPU.
2. Profile (or reason) about bottlenecks.
3. Convert the hotspot into a Manual-style pipelined version.

The tutorial guide is:

- `docs/coding/tutorial.md`

## 8.4 Performance patterns

Common performance patterns show up across kernels:

- 2D tiling by `block_idx` for bandwidth-friendly GM accesses
- staged pipelines: `TLOAD → TEXTRACT/TMOV → TMATMUL → TSTORE`
- overlapping via double buffering (warm-up / steady / drain)

Practical examples:

- GEMM: `kernels/manual/a2a3/gemm_performance/README.md`
- Flash Attention: `kernels/manual/a2a3/flash_atten/README.md`

## 8.5 Debugging

See:

- `docs/coding/debug.md`
