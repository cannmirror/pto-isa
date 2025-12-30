# 5. Synchronization

## 5.1 Why synchronization exists

Many PTO backends implement asynchronous pipelines:

- memory movement and compute may overlap
- multiple functional units can run concurrently

Therefore, a program must enforce ordering between producer/consumer stages.

## 5.2 The TSYNC abstraction

`TSYNC` is the portable synchronization abstraction described in the ISA docs:

- it establishes ordering between two instruction classes (e.g. `TLOAD → TMATMUL`)
- it is typically used to express pipeline edges

See: `docs/isa/TSYNC.md`

## 5.3 Events and flags (backend primitives)

Some implementations expose lower-level event/flag primitives:

- “set flag”: signal completion of a stage
- “wait flag”: block until the signal is observed

These primitives are backend-specific, but the conceptual dependency graph is common.

See: `docs/coding/Event.md`

## 5.4 Rule of thumb (manual kernels)

When writing manual pipelined kernels:

- wait only on **true dependencies**
- structure code as **warm-up → steady state → drain**
- verify correctness first (CPU simulator), then optimize overlap

Reference implementations:

- GEMM: `kernels/manual/a2a3/gemm_performance/README.md`
- Flash Attention: `kernels/manual/a2a3/flash_atten/README.md`
