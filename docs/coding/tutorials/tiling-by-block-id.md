# Tutorial: Split a GlobalTensor into Tiles with `block_idx`

Many PTO kernels use an SPMD-style execution model: the **same kernel** runs on many cores, and each core derives its unique work assignment from its runtime identity.

This tutorial shows how to map a block ID (`get_block_idx()`) to a **tile coordinate**, compute the corresponding **GM pointer offset**, and build `GlobalTensor` + `Tile` objects that operate on that tile.

Related docs:

- Multicore block model: `docs/coding/Block.md`
- Global memory views: `docs/coding/GlobalTensor.md`
- Tile shapes / valid regions: `docs/coding/Tile.md`

## 1. Block IDs: `cid` vs `vid`

Use this convention when sub-core decomposition exists:

```cpp
const std::uint32_t cid = get_block_idx();
const std::uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

- Use `cid` for coarse partitioning across cores.
- Use `vid` when you need a stable “virtual lane id” across sub-blocks.

Most “one block owns one output tile” kernels use `cid` only.

## 2. 2-D tiling (no edge tiles)

Assume a row-major matrix `X[GRows, GCols]`, and a tile shape `(TRows, TCols)` such that:

- `GRows % TRows == 0`
- `GCols % TCols == 0`

Then each block can process exactly one tile:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int GRows, int GCols, int TRows, int TCols>
__global__ AICORE void VecAddTiled(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  constexpr std::uint32_t tiles_per_row = static_cast<std::uint32_t>(GCols / TCols);
  const std::uint32_t cid = get_block_idx();

  const std::uint32_t tile_row = cid / tiles_per_row;
  const std::uint32_t tile_col = cid % tiles_per_row;

  const std::uint64_t base = static_cast<std::uint64_t>(tile_row) * (static_cast<std::uint64_t>(GCols) * TRows) +
                             static_cast<std::uint64_t>(tile_col) * TCols;

  using TileShape = TileShape2D<T, TRows, TCols, Layout::ND>;
  using BaseStride = BaseShape2D<T, GRows, GCols, Layout::ND>;
  using GT = GlobalTensor<T, TileShape, BaseStride, Layout::ND>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols>;

  GT ga(a + base), gb(b + base), gout(out + base);
  TileT ta, tb, tc;

  TLOAD(ta, ga);
  TLOAD(tb, gb);
  TADD(tc, ta, tb);
  TSTORE(gout, tc);
}
```

Launch configuration: set `blockDim = (GRows / TRows) * (GCols / TCols)` so each `cid` maps to a valid tile.

## 3. Edge tiles (dynamic valid region + dynamic GlobalTensor shape)

If `GRows` or `GCols` is not divisible by the tile shape, the last tile on each row/column is an “edge tile”.

The safe pattern is:

1. Compute how many tiles exist: `tiles_rows = ceil_div(GRows, TRows)`, `tiles_cols = ceil_div(GCols, TCols)`.
2. Map `cid` to `(tile_row, tile_col)` in that grid.
3. Guard out-of-range `cid`.
4. Compute `valid_rows`, `valid_cols` for the edge tile.
5. Use:
   - a `Tile` with dynamic valid region, and
   - a `GlobalTensor` with dynamic `(rows, cols)` shape for the tile view.

```cpp
PTO_INLINE std::uint32_t ceil_div_u32(std::uint32_t a, std::uint32_t b) {
  return (a + b - 1) / b;
}

template <typename T, std::uint32_t GRows, std::uint32_t GCols, std::uint32_t TRows, std::uint32_t TCols>
__global__ AICORE void VecAddTiledEdges(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  const std::uint32_t tiles_rows = ceil_div_u32(GRows, TRows);
  const std::uint32_t tiles_cols = ceil_div_u32(GCols, TCols);
  const std::uint32_t num_tiles = tiles_rows * tiles_cols;

  const std::uint32_t cid = get_block_idx();
  if (cid >= num_tiles) return;

  const std::uint32_t tile_row = cid / tiles_cols;
  const std::uint32_t tile_col = cid % tiles_cols;

  const std::uint32_t row0 = tile_row * TRows;
  const std::uint32_t col0 = tile_col * TCols;
  const std::uint32_t valid_rows = (row0 + TRows <= GRows) ? TRows : (GRows - row0);
  const std::uint32_t valid_cols = (col0 + TCols <= GCols) ? TCols : (GCols - col0);

  const std::uint64_t base = static_cast<std::uint64_t>(row0) * GCols + col0;

  using TileShapeDyn = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
  using BaseStride = BaseShape2D<T, GRows, GCols, Layout::ND>;
  using GT = GlobalTensor<T, TileShapeDyn, BaseStride, Layout::ND>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  const TileShapeDyn tile_shape(static_cast<int>(valid_rows), static_cast<int>(valid_cols));
  GT ga(a + base, tile_shape);
  GT gb(b + base, tile_shape);
  GT gout(out + base, tile_shape);

  TileT ta(static_cast<int>(valid_rows), static_cast<int>(valid_cols));
  TileT tb(static_cast<int>(valid_rows), static_cast<int>(valid_cols));
  TileT tc(static_cast<int>(valid_rows), static_cast<int>(valid_cols));

  TLOAD(ta, ga);
  TLOAD(tb, gb);
  TADD(tc, ta, tb);
  TSTORE(gout, tc);
}
```

This pattern ensures the memory ops (`TLOAD`/`TSTORE`) never read/write beyond the valid GM region for edge tiles.

## 4. GEMM-style block mapping (2-D grid)

Many GEMM kernels treat `cid` as a 2-D index into the output tile grid. For example, `kernels/gemm_basic/gemm_basic_kernel.cpp` derives:

```cpp
// Conceptually: cid = nIterIdx * mIter + mIterIdx
const std::uint32_t mIterIdx = get_block_idx() % mIter;
const std::uint32_t nIterIdx = get_block_idx() / mIter;
```

Then compute independent GM pointers for A/B/C blocks:

- `A`: offset by `mIterIdx * singleCoreM * K`
- `B`: offset by `nIterIdx * K * singleCoreN`
- `C`: offset by `mIterIdx * singleCoreM * N + nIterIdx * singleCoreN`

This is the same idea as 2-D tiling; the only difference is that the stride math is derived from GEMM’s `(M,K,N)` dimensions rather than a simple `(rows, cols)` matrix.
