# 2. Execution model

## 2.1 SPMD-style block execution

PTO kernels are typically executed in an SPMD style:

- all participating cores run the same entry function
- each core derives its work assignment from its runtime identity (e.g. `block_idx`)

The portable rule is:

- `block_idx` identifies the **block** (core-level task)
- `subblockid` (if present) identifies a **sub-block instance** within a core

When sub-block decomposition exists, a stable “virtual id” can be constructed:

```cpp
auto cid = get_block_idx();
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

## 2.2 MPMD-style task execution

PTO also supports an MPMD-style execution model.

In **MPMD**, different cores (or groups of cores) can execute **different tile programs** as part of the same overall
tile graph. In the abstract model, the **Device Machine scheduler** assigns work by selecting a tile block and mapping
it onto an available Core Machine.

MPMD is useful when a workload naturally decomposes into multiple stages with different instruction mixes (for example
producer/consumer pipelines or “control-heavy” coordination alongside compute-heavy kernels).

A common portable representation is:

- the scheduler provides a **task id** (or program id) per launched work item
- a single kernel entry function dispatches based on that id, or separate entry points are used

MPMD and SPMD are often combined: each task can still use `block_idx` tiling internally to process its assigned region.

## 2.3 PTO machine abstraction

PTO documentation commonly uses a multi-level abstraction:

- **Core Machine**: executes a single instruction stream and manages on-chip tile state
- **Device Machine**: schedules blocks across multiple cores and handles inter-core dependencies
- **Host Machine**: higher-level orchestration (JIT, caches, graph scheduling), if applicable

The detailed description is in `docs/machine/abstract-machine.md`.

## 2.4 Implications for kernel writers

From a kernel author’s perspective:

- you must partition global tensors into block-local tiles using `block_idx` (and optionally `subblockid`)
- you must respect synchronization requirements when overlapping data movement and compute
- you should prefer regular, contiguous GM access patterns per block for performance

For concrete examples, see:

- `docs/coding/tutorial.md`
- `docs/coding/tutorials/vec-add.md`
