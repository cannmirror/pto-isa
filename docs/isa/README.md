# docs/isa/

This directory contains the PTO ISA instruction reference. Each instruction has its own page describing semantics, constraints, the Auto/Manual forms (if applicable), and examples.

## How to Use

- Start from the ISA guide and notation: `docs/README.md`
- Look up a specific instruction: open `docs/isa/<Instr>.md` (for example: `docs/isa/TADD.md`, `docs/isa/TMATMUL.md`)
- Read by category (recommended for newcomers):
  - Element-wise ops: `docs/isa/Element.md`
  - Tile-scalar ops: `docs/isa/TileScalar.md`
  - Axis reductions / broadcasts: `docs/isa/Axis.md`
  - Matrix multiply family: `docs/isa/Matmul.md`
  - Fixed pipeline / hardware capabilities: `docs/isa/FixPipe.md`
  - Memory (GM ↔ Tile): `docs/isa/Mem.md`
  - Complex ops: `docs/isa/Complex.md`
  - Communication: `docs/isa/Communication.md`
  - Manual / custom patterns: `docs/isa/Manual.md`, `docs/isa/Custom.md`
