# PTO IR (Tile-Level IR) Specification

PTO IR is a tile-level SSA IR designed to model PTO Tile Lib programs above the hardware ISA, while staying close enough to lower into PTO intrinsics and/or target opcodes.

This document defines the intended syntax and semantics of the IR used by the docs in `docs/isa/*` (see each instruction's "IR Syntax" section).

## 1. Goals and Scope

- Tile-centered SSA: tiles are SSA values; ops consume/produce tiles.
- Hardware-friendly but not hardware-exposing: IR can express location/layout/valid/event dependencies but does not require exposing all implementation details.
- Progressive lowering: start with sequential semantics; optionally introduce events and software pipelining.
- Multi-rank tiles are allowed in the type system, but some ops impose rank constraints (e.g., matmul/extract commonly require rank=2).

## 2. Type System

### 2.1 Scalar types

- `index`
- Integer: `i8/i16/i32/i64`
- Float: `f16/f32`, `bf16`, and optional `f8*` flavors depending on target support

### 2.2 Tile type

Syntax:

```ebnf
tile-type ::= 'tile' '<' shape 'x' elem-type (',' tile-type-attrs)? '>'
shape     ::= dim ('x' dim)*
dim       ::= integer-literal | '?'
elem-type ::= f16 | f32 | bf16 | i8 | i16 | i32 | ... | f8e4m3fn | f8e5m2
```

Examples:

- `tile<8xf16>`
- `tile<32x32xf32>`
- `tile<1x4x32x16xf8e4m3fn>`

Note: location/layout/valid are not required to be encoded in the tile type. A common approach is to keep the SSA type lightweight and attach detailed tile info via attributes/operands.

### 2.3 Memref type

PTO IR uses a memref-like type with explicit "space" and (optional) layout metadata:

```text
pto.memref<shape x elem-type, {space = #pto.space<gm|ub|...>, layout = #pto.layout<ND|DN|...>}>
```

Strided variants may appear when canonicalizing views:

```text
pto.memref<32x32xf32, strided<[1024, 1], offset: ?>>
```

## 3. Attributes and Enumerations

### 3.1 Tile location and layout

Used to guide target-specific lowering:

- `#pto.loc<Vec|Mat|Left|Right|Acc|Bias|Scale|...>`
- `#pto.layout<ND|DN|Zz|Zn|Nz|Nn>`

### 3.2 Valid region (mask)

- `valid = [v0, v1, ...]` where each entry is either an integer literal or an SSA `index`.
- If omitted, `valid == shape`.

## 4. Operation Set (Core)

All tile ops are defined in the `pto.tile.*` namespace. Ops are provided in:

- a synchronous form (sequential semantics), and
- an optional asynchronous form that returns an event token and supports `wait(...)`.

### 4.1 Scalar and index ops

```mlir
%c32 = pto.constant 32 : index
%cz  = pto.constant 0.0 : f32
%0 = pto.muli %arg3, %c32 : index
```

### 4.2 Memref view ops (optional)

`pto.memref.subview` creates a view that should be canonicalized early:

```mlir
%sv = pto.memref.subview %arg0[%o0, %o1] [32, 32] [1, 1]
    : pto.memref<1024x1024xf32>
   to pto.memref<32x32xf32, strided<[1024, 1], offset: ?>>
```

### 4.3 Tile load/store

#### `pto.tile.load`

```mlir
%t0 = pto.tile.load %sv[%c0, %c0]
    : pto.memref<32x32xf32, strided<[1024, 1], offset: ?>>, tile<32x32xf32>
```

Optional attributes (examples):

- `{loc = #pto.loc<Vec|Mat|...>}`
- `{layout = #pto.layout<...>}`
- `{valid = [...]}` (dynamic mask)

#### `pto.tile.store`

```mlir
pto.tile.store %t1, %sv_out[%c0, %c0]
    : pto.memref<32x32xf32, strided<[1024, 1], offset: ?>>, tile<32x32xf32>
```

### 4.4 Elementwise tile ops

#### `pto.tile.add` (maps to `TADD`)

```mlir
%t2 = pto.tile.add %t0, %t1 : tile<32x32xf32>
```

Verifier rules (IR-level):

- Tile rank/shape/element type must match.
- If `valid` is present: `0 < valid[d] <= shape[d]` for each dimension.

Other elementwise ops follow the same pattern:

`pto.tile.sub/mul/div/min/max/exp/sqrt/rsqrt/...`

## 5. Asynchrony and Events

To express dependencies without `TSYNC`, PTO IR uses SSA event tokens.

### 5.1 Event type

```text
!pto.event<producer = #pto.op<TLOAD>, scope = #pto.scope<intra|inter>, eid = 0>
```

Key points:

- The token carries the producer opcode.
- The consumer opcode is implied by the op that uses the token (lowering can derive the correct set/wait).

### 5.2 Asynchronous op form

Potentially asynchronous ops (load/store/vec/matmul/fixpipe) can provide:

- an optional `wait(%e0, %e1, ...)` clause, and
- an event result `!pto.event<producer = #pto.op<...>>`.

Example (load/add/store pipeline):

```mlir
%t0, %e0 = pto.tile.load %sv0[%c0, %c0] wait(%e_prev)
    : pto.memref<32x32xf32, ...>, tile<32x32xf32>, !pto.event<producer = #pto.op<TLOAD>>

%t1, %e1 = pto.tile.load %sv1[%c0, %c0] wait(%e_prev)
    : pto.memref<32x32xf32, ...>, tile<32x32xf32>, !pto.event<producer = #pto.op<TLOAD>>

%t2, %e2 = pto.tile.add %t0, %t1 wait(%e0, %e1)
    : tile<32x32xf32>, !pto.event<producer = #pto.op<TADD>>

%e3 = pto.tile.store %t2, %sv2[%c0, %c0] wait(%e2)
    : pto.memref<32x32xf32, ...>, tile<32x32xf32>, !pto.event<producer = #pto.op<TSTORE_VEC>>
```

## 6. Fixpipe and Matmul Extensions

### 6.1 `pto.tile.mov.*` (maps to `TMOV` family)

The IR recommends splitting `TMOV` into multiple ops to avoid large attribute sets:

```mlir
%left  = pto.tile.mov.m2l %mat : tile<...> -> tile<...>
%right = pto.tile.mov.m2r %mat : tile<...> -> tile<...>
%bias  = pto.tile.mov.m2b %mat : tile<...> -> tile<...>
%scale = pto.tile.mov.m2s %mat : tile<...> -> tile<...>
%vec   = pto.tile.mov.a2v %acc : tile<...> -> tile<...>
%v1    = pto.tile.mov.v2v %v0  : tile<...> -> tile<...>
```

### 6.2 `pto.tile.extract` (maps to `TEXTRACT`)

```mlir
%dst = pto.tile.extract %src[%r0, %r1]
    : tile<SrcShape x Ts, #pto.tile_info<loc=Mat, layout=Ls>>
   -> tile<DstShape x Ts, #pto.tile_info<loc=Left|Right, layout=Ld>>
```

### 6.3 `pto.tile.matmul*` (maps to `TMATMUL*`)

```mlir
%acc = pto.tile.matmul %a, %b
    : tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>
   -> tile<MxNxTc, #pto.tile_info<loc=Acc,   layout=Lc>>

%acc1 = pto.tile.matmul.acc %acc0, %a, %b
    : tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>,
      tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>
   -> tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>

%acc2 = pto.tile.matmul.bias %a, %b, %bias
    : tile<MxKxTa, #pto.tile_info<loc=Left,  layout=La>>,
      tile<KxNxTb, #pto.tile_info<loc=Right, layout=Lb>>,
      tile<1xNxTb2, #pto.tile_info<loc=Bias, layout=Lbias>>
   -> tile<MxNxTc, #pto.tile_info<loc=Acc, layout=Lc>>
```

## 7. Verification and Target Legality

Recommended layering:

- IR-level verifier: SSA/type checks, shape/rank constraints, bounds for extract, valid range checks.
- Target legality: alignment, tile size limits, supported dtype/layout combinations, and target-specific constraints.

Example target annotation:

```mlir
module attributes { pto.target = #pto.target<a3> } { ... }
```

## 8. Suggested Pass Pipeline

1. Canonicalize: fold constants and remove dead code.
2. Canonicalize memref views: rewrite into base+offset+stride early.
3. Infer tile info: fill default loc/layout/valid as needed.
4. (Optional) Insert events: make critical async boundaries explicit.
5. (Optional) Software pipeline: generate ping-pong/event scheduling.
6. Bufferize: map SSA tiles to physical tile storage and insert `TASSIGN` as needed.
7. Lower to PTO ISA / C++ intrinsics.

