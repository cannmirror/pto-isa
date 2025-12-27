# PTO ISA Architectural Manual

## Preface

This manual describes the **PTO ISA** (Parallel Tile Operation Instruction Set Architecture) and its associated execution and programming model.

The goal is to provide an ISA-level reference in a consistent style:

- stable terminology and precise definitions
- clear operand and shape rules
- cross-references to detailed instruction pages

This manual is intentionally *not* a full “one-page-per-instruction” encyclopedia. For individual instruction semantics and constraints, see `docs/isa/README.md`.

## Audience

This manual is written for:

- kernel authors writing PTO intrinsics/assembly
- compiler/runtime developers lowering PTO programs
- CPU simulator developers validating instruction semantics

## Conformance

Unless a section explicitly says otherwise:

- “**must**” indicates a mandatory constraint (violations are errors).
- “**should**” indicates a recommended practice (violations may still work but are discouraged).
- “**may**” indicates an optional behavior/feature.

## Document map (reading order)

1. Overview: `manual/01-overview.md`
2. Execution model: `manual/02-machine-model.md`
3. State and types: `manual/03-state-and-types.md`
4. Tiles and GlobalTensor: `manual/04-tiles-and-globaltensor.md`
5. Synchronization: `manual/05-synchronization.md`
6. PTO assembly: `manual/06-assembly.md`
7. Instruction set structure: `manual/07-instructions.md`
8. Programming guide (Auto/Manual patterns): `manual/08-programming.md`

