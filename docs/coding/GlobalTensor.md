# GlobalTensor Programming Model

`pto::GlobalTensor` models a tensor stored in global memory (GM). It is a lightweight wrapper around a `__gm__` pointer plus a 5-D shape and stride description used by memory instructions such as `TLOAD` and `TSTORE`.

All identifiers in this document refer to definitions in `include/pto/common/pto_tile.hpp` unless noted otherwise.

## GlobalTensor type

```cpp
template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor;
```

- `Element_`: scalar element type stored in GM.
- `Shape_`: a `pto::Shape<...>` describing up to 5 dimensions.
- `Stride_`: a `pto::Stride<...>` describing up to 5 per-dimension strides (in elements).
- `Layout_`: `pto::Layout` (`ND`, `DN`, `NZ`, ...). This guides lowering and target-specific paths.

The GM pointer type is `GlobalTensor::DType`, which is `__gm__ Element_`.

## Shapes and strides (5-D)

### `pto::Shape`

`pto::Shape<N1, N2, N3, N4, N5>` stores 5 integers. Each template parameter can be a compile-time constant or `pto::DYNAMIC` (`-1`).

- Static dimensions are carried in the type via `Shape::staticShape[dim]`.
- Dynamic dimensions are stored in the runtime `Shape::shape[dim]` and are populated by the `Shape(...)` constructor.

### `pto::Stride`

`pto::Stride<S1, S2, S3, S4, S5>` follows the same pattern as `Shape`, but stores strides.

Strides are expressed in **elements**, not bytes.

### `GlobalTensor` construction and access

`GlobalTensor` stores a pointer plus runtime shape/stride values for dynamic dimensions:

```cpp
GTensor t(ptr, shape, stride);
auto* p = t.data();
int d4 = t.GetShape(pto::GlobalTensorDim::DIM_4);
int s4 = t.GetStride(pto::GlobalTensorDim::DIM_4);
```

For fully-static tensors you can also query compile-time values:

```cpp
constexpr int cols = GTensor::GetShape<pto::GlobalTensorDim::DIM_4>();
```

## 2-D helpers used by examples

Two helper families are commonly used for 2-D tensors:

- `pto::TileShape2D<T, rows, cols, layout>`: produces a `pto::Shape<1,1,1,rows,cols>`.
- `pto::BaseShape2D<T, rows, cols, layout>`: produces a `pto::Stride<...>` suitable for a base 2-D view.

Despite its name, `BaseShape2D` is a **stride** helper (it derives from `pto::Stride`).

## Address binding (`TASSIGN`)

`TASSIGN(globalTensor, ptr)` sets the underlying GM pointer of a `GlobalTensor`. The pointer type must match `GlobalTensor::DType` (enforced by `static_assert` in `TASSIGN_IMPL`).

## Minimal example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example(__gm__ float* in, __gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  GT gout(out);

  TileT t;
  TLOAD(t, gin);
  TSTORE(gout, t);
}
```

