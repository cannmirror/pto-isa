# PTO-AS (PTO Assembly) Specification

PTO-AS is a textual, instruction-centric assembly format for PTO Tile Lib. It is designed to be:

- close to the PTO instruction set (`TADD`, `TLOAD`, `TMATMUL`, ...),
- readable and easy to diff (one instruction per line),
- compatible with MLIR tooling (SSA value naming, MLIR-like type spellings, MLIR bytecode as the interchange format).

PTO-AS is designed to be consumed/produced by an MLIR-based assembler/disassembler.

## 1. High-Level Form

A PTO-AS program is a list of statements. The most common statement is an instruction:

```text
%dst = tadd %src0, %src1 : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> !pto.tile<32x32xf32>;
```

PTO-AS uses SSA-like value names (`%dst`, `%src0`) to stay close to MLIR’s assembly conventions; this keeps the
format deterministic and makes it easy to round-trip through MLIR bytecode.

PTO-AS is a synchronous, line-ordered format: there is no `wait(...)` clause and no implicit event result. If a program
needs to model an explicit dependency, it uses an explicit instruction (for example `tsync`) with event operands.

Operands may also include indexed forms (commonly used by memory ops):

```text
%t0 = tload %sv[%c0, %c1] : (!pto.memref<...>, index, index) -> !pto.tile<...>;
```

Type signatures (`: ...`) are recommended for readability but may be omitted when the types are unambiguous in context.

## 2. Types

PTO-AS uses MLIR-like type spellings:

- Tile values: `!pto.tile<...>` (opaque)
- Global memory / views: `!pto.memref<...>` (opaque)
- Events: `!pto.event` (opaque)
- Scalars: MLIR builtin types like `index`, `i32`, `f32`

The assembler treats these as *opaque* types; they are carried through bytecode but not semantically verified unless a
target-specific verifier is introduced later.

## 3. Attributes

Instruction modifiers that are not positional operands (e.g., compare modes) are written as an MLIR-style attribute
dictionary:

```text
%mask = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

## 4. Directives

PTO-AS supports a small set of non-instruction directives for declaring external inputs and constants.

Argument declaration (introduces an SSA value):

```text
.arg %a : !pto.tile<16x16xf16>;
```

Event arguments (when modeling a dependency explicitly):

```text
.arg %e0 : !pto.event;
```

Constant declaration (introduces an SSA value):

```text
.const %c0 = 0 : index;
```

## 5. Grammar

The normative grammar is provided in:

- `docs/grammar/PTO-AS.bnf`
