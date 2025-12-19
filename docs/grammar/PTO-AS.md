# PTO-AS (PTO Assembly) Specification

PTO-AS is a textual, instruction-centric assembly format for PTO Tile Lib. It is designed to be:

- close to the PTO instruction set (`TADD`, `TLOAD`, `TMATMUL`, ...),
- readable and easy to diff (one instruction per line),
- compatible with MLIR tooling (SSA value naming, MLIR-like type spellings, MLIR bytecode as the interchange format).

PTO-AS is designed to be consumed/produced by an MLIR-based assembler/disassembler (tooling may be added in a future update).

## 1. High-Level Form

A PTO-AS program is a list of statements. The most common statement is an instruction:

```text
%dst = tadd %src0, %src1 : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> !pto.tile<32x32xf32>;
```

PTO-AS uses SSA-like value names (`%dst`, `%src0`) to stay close to MLIR’s assembly conventions; this keeps the
format deterministic and makes it easy to round-trip through MLIR bytecode.

Operands may also include PTX-like “indexed” forms (commonly used by memory ops):

```text
%t0 = tload %sv[%c0, %c1] : (!pto.memref<...>, index, index) -> !pto.tile<...>;
```

## 2. Types

PTO-AS uses MLIR-like type spellings:

- Tile values: `!pto.tile<...>` (opaque)
- Global memory / views: `!pto.memref<...>` (opaque)
- Events: `!pto.event` (opaque)
- Scalars: MLIR builtin types like `index`, `i32`, `f32`

The assembler treats these as *opaque* types; they are carried through bytecode but not semantically verified unless a
target-specific verifier is introduced later.

## 3. Wait / Event Convention

To model hardware-style dependencies without embedding full IR, PTO-AS supports:

- a `wait(%e0, %e1, ...)` clause on any instruction, and
- an optional event result as an additional SSA result.

Example:

```text
%dst, %e = tadd %src0, %src1 wait(%e0, %e1)
    : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> (!pto.tile<32x32xf32>, !pto.event);
```

## 4. Attributes

Instruction modifiers that are not positional operands (e.g., compare modes) are written as an MLIR-style attribute
dictionary:

```text
%mask = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

## 5. Directives (Optional)

PTO-AS supports a small set of non-instruction directives for declaring external inputs and constants.

Argument declaration (introduces an SSA value):

```text
.arg %a : !pto.tile<16x16xf16>;
```

Constant declaration (introduces an SSA value):

```text
.const %c0 = 0 : index;
```

## 6. Grammar (BNF)

The normative grammar is provided in:

- `docs/grammar/as/PTO-AS.bnf`
