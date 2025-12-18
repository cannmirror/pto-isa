# PTO ISA Guide

This directory documents the PTO ISA (Instruction Set Architecture) used by PTO Tile Lib. It explains instruction naming, common notation, and how to navigate the per-instruction reference pages.

## Naming and Notation

- **Tile**: Fundamental data type for small tensors (e.g., `MatTile`, `LeftTile`, `RightTile`, `BiasTile`, `AccumulationTile`, `VecTile`).
- **GlobalTensor**: A tensor stored in global memory (GM). `TLOAD`/`TSTORE` move data between GM and Tiles.
- **`%R`**: A scalar immediate register. Fields like `cmpMode` and `rmode` are instruction modifiers.
- **Shape and alignment**: Enforced by a combination of compile-time constraints and runtime assertions; invalid usage should fail fast.

## Where to Start

- ISA overview: `docs/PTOISA.md`
- Instruction index: `docs/isa/README.md`
- Common conventions: `docs/isa/conventions.md`
- PTO IR syntax reference: `docs/ir/PTO-IR.md`
- Getting started (recommended: run on CPU first): `docs/getting-started.md`
- Implementation and extension notes: `docs/coding/README.md`

## Documentation Layout

- `docs/isa/`: Instruction reference (one file per instruction, plus category pages)
- `docs/ir/`: PTO IR syntax and semantics reference used by instruction docs
- `docs/coding/`: Developer notes for extending PTO Tile Lib
