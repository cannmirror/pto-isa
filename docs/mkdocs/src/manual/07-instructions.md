# 7. Instruction set

## 7.1 Instruction reference

The canonical instruction reference is:

- `docs/isa/README.md`

Each instruction page defines:

- assembly syntax
- intrinsic API shape (when applicable)
- operand constraints (types, shapes, layouts, locations)
- semantic definition over the valid region
- examples and CPU simulator notes

## 7.2 Category overview

This manual uses the following categories:

### Data movement

- `TLOAD`, `TSTORE`
- `TASSIGN`
- `TEXTRACT`, `TMOV`, `TTRANS`, `TRESHAPE`

### Elementwise and scalar-tile ops

- arithmetic: `TADD`, `TSUB`, `TMUL`, `TDIV`, …
- math: `TEXP`, `TLOG`, `TSQRT`, `TRSQRT`, `TRECIP`, …
- bitwise: `TAND`, `TOR`, `TXOR`, shifts, …

### Reductions and expands

- row reductions: `TROWMAX`, `TROWMIN`, `TROWSUM`
- col reductions: `TCOLMAX`, `TCOLMIN`, `TCOLSUM`
- expand/broadcast: `TROWEXPAND`, `TCOLEXPAND`, and friends

### Compare / select

- compare: `TCMP`, `TCMPS`
- select: `TSEL`, `TSELS`

### Gather / scatter

- tile gather/scatter: `TGATHER`, `TSCATTER`
- memory gather/scatter: `MGATHER`, `MSCATTER`

### Matrix multiply

- `TMATMUL` and accumulation variants

### Synchronization

- `TSYNC`

## 7.3 Common semantic rule

Unless an instruction states otherwise:

- results are defined for indices in the operand valid region(s)
- undefined/out-of-mask elements are not part of the architectural contract

If you rely on out-of-range values, make them explicit with a defined padding/fill operation.
