# Block (Multicore) Programming Model

Similar to SPMD (Single Program Multiple Data) models such as Triton and cuTile, PTO parallelizes execution by **partitioning the workload into blocks** and dispatching those blocks across multiple AI Cores.

All participating cores execute the **same kernel instruction stream**. Per-core behavior diverges only through a core’s runtime identity (for example, `block_idx`), which determines which **GlobalTensor slice** and which **Tiles** each core operates on.

In the PTO abstract machine model (`docs/machine/abstract-machine.md`):

- A **block** is typically one **tile program** (a short instruction sequence operating on a small set of tiles and GM views).
- A **Core Machine** executes one block at a time.
- A **Device Machine** schedules blocks onto available Core Machines and tracks global-memory dependencies.

## Block identity: `cid` vs `vid`

On many targets, a kernel can query its identity via builtins. The most common pattern is:

```cpp
auto cid = get_block_idx();
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

- `cid` is the **core-level block ID** used for coarse task partitioning.
- `vid` is a **sub-block virtual ID** that remains stable if a core is further subdivided into multiple execution instances.

This convention is useful because AI Cores may be organized into different execution roles (for example, Vector vs Cube), and the **Cube:Vector composition can vary across generations**. When sub-core decomposition exists, `vid` gives a consistent “SPMD lane ID” across those variations.

## Typical work partitioning pattern

A common PTO operator structure is:

1. Every core enters the same entry function with identical arguments (GM pointers, scalars).
2. Each core computes `(cid, vid)` and maps it to a tile coordinate or a GM offset.
3. The core loads its inputs (`TLOAD`), computes on tiles, and writes outputs (`TSTORE`).

Example: map a linear `cid` to a 2-D tile grid:

```cpp
const std::uint32_t cid = get_block_idx();
const std::uint32_t tiles_per_row = (GCols + TCols - 1) / TCols;
const std::uint32_t tile_row = cid / tiles_per_row;
const std::uint32_t tile_col = cid % tiles_per_row;
```

From `(tile_row, tile_col)`, kernels typically compute GM offsets and construct `GlobalTensor` views for the tile that core owns.

## Safety and determinism guidelines

- Prefer **disjoint output tiles per `cid`/`vid`**. If two blocks write the same GM region, results are undefined unless you use an atomic write mode where supported (see `docs/isa/TSTORE.md`).
- `TSYNC`/events order operations **within a core** (and across pipeline classes). They do not, by themselves, define cross-core ordering.
- Cross-core ordering and visibility are generally expressed via:
  - Global-memory producer/consumer structure (write then later read), as defined by the runtime/driver contract; and/or
  - Explicit device-level synchronization provided by the platform runtime (outside the PTO ISA scope).

## Where to read next

- PTO-Auto vs PTO-Manual: `docs/coding/ProgrammingModel.md`
- Tutorial: map `block_idx` to per-tile `GlobalTensor` views: `docs/coding/tutorials/tiling-by-block-id.md`
- Tiles and valid-region semantics: `docs/coding/Tile.md`
- GlobalTensor (GM) views and layouts: `docs/coding/GlobalTensor.md`
- Event tokens and `TSYNC`: `docs/coding/Event.md`
