# 4. Tiles and GlobalTensor

## 4.1 Tile as the fundamental execution unit

Most PTO instructions operate on a Tile and treat it as a 2D array:

```
T[r, c]   where 0 ≤ r < R, 0 ≤ c < C
```

When a Tile has a valid region `[Rv, Cv]`, the ISA meaning is defined only for:

```
0 ≤ r < Rv, 0 ≤ c < Cv
```

## 4.2 Location and intent

Tile `Location` encodes intent and constraints. Typical locations include:

- `Vec`: vector/elementwise/reduction operations
- `Mat`: general matrix tile used for movement/reshape/transpose
- `Left`, `Right`, `Acc`: cube/matrix-multiply operands
- `Bias`, `Scale`: auxiliary matrix data for fused operations

Location constraints are instruction-specific; see the corresponding instruction page.

## 4.3 Moving data: GM ↔ Tile

Canonical data movement instructions:

- `TLOAD(tile, global)`: loads a rectangular region from GM into a tile
- `TSTORE(global, tile)`: stores a rectangular region from a tile back to GM

Other movement/shape/layout operations:

- `TEXTRACT`: slice a sub-rectangle (or window) into a tile
- `TMOV`: convert between tile locations/layouts (often for cube preparation)
- `TTRANS`: transpose
- `TRESHAPE`: reinterpret layout/shape metadata (with constraints)

## 4.4 Practical mapping rule (tiling by block id)

Most kernels follow the same mapping:

- interpret GM as an `M×N` matrix
- choose tile size `TR×TC`
- compute tile row/col from `block_idx`

For a concrete worked example, see `docs/coding/tutorials/vec-add.md`.

