/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TILE_HPP
#define PTO_TILE_HPP

#include "pto/common/memory.hpp"
#include <pto/common/type.hpp>
#include <pto/common/constants.hpp>
#ifdef __CPU_SIM
#include <iomanip>
#endif

namespace pto {

enum class Layout {
    ND, // ND RowMajor
    DN, // DN ColMajor
    NZ, // NZ for cube
    SCALE,
    MAX,
};
namespace GlobalTensorDim {
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;
constexpr int DIM_4 = 4;
constexpr int TOTAL_DIM = 5;
} // namespace GlobalTensorDim

constexpr int DYNAMIC = -1;

template <int N1 = DYNAMIC, int N2 = DYNAMIC, int N3 = DYNAMIC, int N4 = DYNAMIC, int N5 = DYNAMIC>
struct Shape {
    static constexpr int staticShape[5] = {N1, N2, N3, N4, N5};
    __aicore__ PTO_INLINE Shape(int n1, int n2, int n3, int n4, int n5)
    {
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = n1;
        if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = n2;
        if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = n3;
        if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = n4;
        if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = n5;
    }

    __aicore__ PTO_INLINE Shape() {
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = 1;
        if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = 1;
        if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = 1;
        if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = 1;
        if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = 1;
    }

    __aicore__ PTO_INLINE Shape(int n) {
        static_assert((N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_1,
            "1-parameter constructors is only applicable to Stride with 1 dynamic dimension.");
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = n;
        else if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = n;
        else if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = n;
        else if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = n;
        else if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = n;
    }

    __aicore__ PTO_INLINE Shape(int n1, int n2) {
        static_assert((N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_2,
            "2-parameter constructors is only applicable to Stride with 2 dynamic dimension.");

        int idx = 0;
        const int vals[] = {n1, n2};
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = vals[idx++];
    }

    __aicore__ PTO_INLINE Shape(int n1, int n2, int n3) {
        static_assert((N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_3,
            "3-parameter constructors is only applicable to Stride with 3 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3};
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = vals[idx++];
    }

    __aicore__ PTO_INLINE Shape(int n1, int n2, int n3, int n4) {
        static_assert((N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_4,
            "4-parameter constructors is only applicable to Stride with 4 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3, n4};
        if constexpr (N1 == DYNAMIC) shape[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (N2 == DYNAMIC) shape[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (N3 == DYNAMIC) shape[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (N4 == DYNAMIC) shape[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (N5 == DYNAMIC) shape[GlobalTensorDim::DIM_4] = vals[idx++];
    }

public:
    int shape[GlobalTensorDim::TOTAL_DIM] = {1};
};

template <int SN1 = DYNAMIC, int SN2 = DYNAMIC, int SN3 = DYNAMIC, int SN4 = DYNAMIC, int SN5 = DYNAMIC>
struct Stride {
    static constexpr int staticStride[GlobalTensorDim::TOTAL_DIM] = {SN1, SN2, SN3, SN4, SN5};
    __aicore__ PTO_INLINE Stride(int n1, int n2, int n3, int n4, int n5)
    {
        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = n1;
        if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = n2;
        if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = n3;
        if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = n4;
        if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = n5;
    }

    __aicore__ PTO_INLINE Stride() {
        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = 1;
        if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = 1;
        if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = 1;
        if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = 1;
        if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = 1;
    }

    __aicore__ PTO_INLINE Stride(int n) {
        static_assert((SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_1,
            "1-parameter constructors is only applicable to Stride with 1 dynamic dimension.");

        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = n;
        else if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = n;
        else if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = n;
        else if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = n;
        else if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = n;
    }

    __aicore__ PTO_INLINE Stride(int n1, int n2) {
        static_assert((SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_2,
            "2-parameter constructors is only applicable to Stride with 2 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2};
        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = vals[idx++];
    }

    __aicore__ PTO_INLINE Stride(int n1, int n2, int n3) {
        static_assert((SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_3,
            "3-parameter constructors is only applicable to Stride with 3 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3};
        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = vals[idx++];
    }

    __aicore__ PTO_INLINE Stride(int n1, int n2, int n3, int n4) {
        static_assert((SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) ==
                          GlobalTensorDim::DIM_4,
            "4-parameter constructors is only applicable to Stride with 4 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3, n4};
        if constexpr (SN1 == DYNAMIC) stride[GlobalTensorDim::DIM_0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC) stride[GlobalTensorDim::DIM_1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC) stride[GlobalTensorDim::DIM_2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC) stride[GlobalTensorDim::DIM_3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC) stride[GlobalTensorDim::DIM_4] = vals[idx++];
    }

public:
    int stride[GlobalTensorDim::TOTAL_DIM] = {1};
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor {
    using Shape = Shape_;
    using Stride = Stride_;
    using DType = __gm__ Element_;
    static constexpr Layout layout = Layout_;

    static const Shape defaultShape;
    static const Stride defaultStride;

    static constexpr int staticShape[GlobalTensorDim::TOTAL_DIM] = {Shape::staticShape[GlobalTensorDim::DIM_0],
        Shape::staticShape[GlobalTensorDim::DIM_1], Shape::staticShape[GlobalTensorDim::DIM_2],
        Shape::staticShape[GlobalTensorDim::DIM_3], Shape::staticShape[GlobalTensorDim::DIM_4]};
    static constexpr int staticStride[GlobalTensorDim::TOTAL_DIM] = {Stride::staticStride[GlobalTensorDim::DIM_0],
        Stride::staticStride[GlobalTensorDim::DIM_1], Stride::staticStride[GlobalTensorDim::DIM_2],
        Stride::staticStride[GlobalTensorDim::DIM_3], Stride::staticStride[GlobalTensorDim::DIM_4]};
    __aicore__ PTO_INLINE GlobalTensor(
        DType *data, const Shape &shape = defaultShape, const Stride &stride = defaultStride)
    {
        data_ = data;

        if constexpr (staticShape[GlobalTensorDim::DIM_0] == DYNAMIC) {
            shape_.shape[GlobalTensorDim::DIM_0] = shape.shape[GlobalTensorDim::DIM_0];
        }
        if constexpr (staticShape[GlobalTensorDim::DIM_1] == DYNAMIC) {
            shape_.shape[GlobalTensorDim::DIM_1] = shape.shape[GlobalTensorDim::DIM_1];
        }
        if constexpr (staticShape[GlobalTensorDim::DIM_2] == DYNAMIC) {
            shape_.shape[GlobalTensorDim::DIM_2] = shape.shape[GlobalTensorDim::DIM_2];
        }
        if constexpr (staticShape[GlobalTensorDim::DIM_3] == DYNAMIC) {
            shape_.shape[GlobalTensorDim::DIM_3] = shape.shape[GlobalTensorDim::DIM_3];
        }
        if constexpr (staticShape[GlobalTensorDim::DIM_4] == DYNAMIC) {
            shape_.shape[GlobalTensorDim::DIM_4] = shape.shape[GlobalTensorDim::DIM_4];
        }

        if constexpr (staticStride[GlobalTensorDim::DIM_0] == DYNAMIC) {
            stride_.stride[GlobalTensorDim::DIM_0] = stride.stride[GlobalTensorDim::DIM_0];
        }
        if constexpr (staticStride[GlobalTensorDim::DIM_1] == DYNAMIC) {
            stride_.stride[GlobalTensorDim::DIM_1] = stride.stride[GlobalTensorDim::DIM_1];
        }
        if constexpr (staticStride[GlobalTensorDim::DIM_2] == DYNAMIC) {
            stride_.stride[GlobalTensorDim::DIM_2] = stride.stride[GlobalTensorDim::DIM_2];
        }
        if constexpr (staticStride[GlobalTensorDim::DIM_3] == DYNAMIC) {
            stride_.stride[GlobalTensorDim::DIM_3] = stride.stride[GlobalTensorDim::DIM_3];
        }
        if constexpr (staticStride[GlobalTensorDim::DIM_4] == DYNAMIC) {
            stride_.stride[GlobalTensorDim::DIM_4] = stride.stride[GlobalTensorDim::DIM_4];
        }
    }

    __aicore__ PTO_INLINE int GetShape(const int dim)
    {
        switch (dim) {
            case GlobalTensorDim::DIM_0: return GetShapeSize<staticShape[GlobalTensorDim::DIM_0]>(dim);
            case GlobalTensorDim::DIM_1: return GetShapeSize<staticShape[GlobalTensorDim::DIM_1]>(dim);
            case GlobalTensorDim::DIM_2: return GetShapeSize<staticShape[GlobalTensorDim::DIM_2]>(dim);
            case GlobalTensorDim::DIM_3: return GetShapeSize<staticShape[GlobalTensorDim::DIM_3]>(dim);
            case GlobalTensorDim::DIM_4: return GetShapeSize<staticShape[GlobalTensorDim::DIM_4]>(dim);
            default: return -1;
        }
    }

    __aicore__ PTO_INLINE int GetStride(const int dim)
    {
        switch (dim) {
            case GlobalTensorDim::DIM_0: return GetStrideSize<staticStride[GlobalTensorDim::DIM_0]>(dim);
            case GlobalTensorDim::DIM_1: return GetStrideSize<staticStride[GlobalTensorDim::DIM_1]>(dim);
            case GlobalTensorDim::DIM_2: return GetStrideSize<staticStride[GlobalTensorDim::DIM_2]>(dim);
            case GlobalTensorDim::DIM_3: return GetStrideSize<staticStride[GlobalTensorDim::DIM_3]>(dim);
            case GlobalTensorDim::DIM_4: return GetStrideSize<staticStride[GlobalTensorDim::DIM_4]>(dim);
            default: return -1;
        }
    }

    template <int dim>
    __aicore__ static constexpr int GetShape()
    {
        static_assert(dim >= GlobalTensorDim::DIM_0 && dim < GlobalTensorDim::TOTAL_DIM, "only support get dim(0-4)");
        if constexpr (dim == GlobalTensorDim::DIM_0) {
            static_assert(staticShape[GlobalTensorDim::DIM_0] != DYNAMIC,
                "dim 0 is dynamic, cannot be obtained using the template interface.");
            return staticShape[GlobalTensorDim::DIM_0];
        }
        if constexpr (dim == GlobalTensorDim::DIM_1) {
            static_assert(staticShape[GlobalTensorDim::DIM_1] != DYNAMIC,
                "dim 1 is dynamic, cannot be obtained using the template interface.");
            return staticShape[GlobalTensorDim::DIM_1];
        }
        if constexpr (dim == GlobalTensorDim::DIM_2) {
            static_assert(staticShape[GlobalTensorDim::DIM_2] != DYNAMIC,
                "dim 2 is dynamic, cannot be obtained using the template interface.");
            return staticShape[GlobalTensorDim::DIM_2];
        }
        if constexpr (dim == GlobalTensorDim::DIM_3) {
            static_assert(staticShape[GlobalTensorDim::DIM_3] != DYNAMIC,
                "dim 3 is dynamic, cannot be obtained using the template interface.");
            return staticShape[GlobalTensorDim::DIM_3];
        }
        if constexpr (dim == GlobalTensorDim::DIM_4) {
            static_assert(staticShape[GlobalTensorDim::DIM_4] != DYNAMIC,
                "dim 4 is dynamic, cannot be obtained using the template interface.");
            return staticShape[GlobalTensorDim::DIM_4];
        }
        return -1;
    }

    template <int dim>
    __aicore__ static constexpr int GetStride()
    {
        static_assert(dim >= GlobalTensorDim::DIM_0 && dim < GlobalTensorDim::TOTAL_DIM, "only support get dim(0-4)");
        if constexpr (dim == GlobalTensorDim::DIM_0) {
            static_assert(staticStride[GlobalTensorDim::DIM_0] != DYNAMIC,
                "dim 0 is dynamic, cannot be obtained using the template interface.");
            return staticStride[GlobalTensorDim::DIM_0];
        }
        if constexpr (dim == GlobalTensorDim::DIM_1) {
            static_assert(staticStride[GlobalTensorDim::DIM_1] != DYNAMIC,
                "dim 1 is dynamic, cannot be obtained using the template interface.");
            return staticStride[GlobalTensorDim::DIM_1];
        }
        if constexpr (dim == GlobalTensorDim::DIM_2) {
            static_assert(staticStride[GlobalTensorDim::DIM_2] != DYNAMIC,
                "dim 2 is dynamic, cannot be obtained using the template interface.");
            return staticStride[GlobalTensorDim::DIM_2];
        }
        if constexpr (dim == GlobalTensorDim::DIM_3) {
            static_assert(staticStride[GlobalTensorDim::DIM_3] != DYNAMIC,
                "dim 3 is dynamic, cannot be obtained using the template interface.");
            return staticStride[GlobalTensorDim::DIM_3];
        }
        if constexpr (dim == GlobalTensorDim::DIM_4) {
            static_assert(staticStride[GlobalTensorDim::DIM_4] != DYNAMIC,
                "dim 4 is dynamic, cannot be obtained using the template interface.");
            return staticStride[GlobalTensorDim::DIM_4];
        }
        return -1;
    }

    __aicore__ DType *data()
    {
        return data_;
    }

private:
    template <int StaticShape>
    __aicore__ PTO_INLINE int GetShapeSize(const int dim)
    {
        if constexpr (StaticShape == DYNAMIC) {
            return shape_.shape[dim];
        } else {
            return StaticShape;
        }
    }

    template <int StaticStride>
    __aicore__ PTO_INLINE int GetStrideSize(const int dim)
    {
        if constexpr (StaticStride == DYNAMIC) {
            return stride_.stride[dim];
        } else {
            return StaticStride;
        }
    }

    DType *data_;
    Shape shape_ = defaultShape;
    Stride stride_ = defaultStride;
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Shape
GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultShape{1, 1, 1, 1, 1};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Stride
GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultStride{1, 1, 1, 1, 1};

template <typename T, int rows = DYNAMIC, int cols = DYNAMIC, Layout Layout_ = Layout::ND>
struct TileShape2D;

template <typename T, int cols>
constexpr int GetTileShape2DNZCols()
{
    if constexpr (cols == DYNAMIC) {
        return DYNAMIC;
    } else {
        return static_cast<int>(cols / (C0_SIZE_BYTE / sizeof(T)));
    }
}

template <typename T, int rows>
constexpr int GetTileShape2DNZRows()
{
    if constexpr (rows == DYNAMIC) {
        return DYNAMIC;
    } else {
        return static_cast<int>(rows / FRACTAL_NZ_ROW);
    }
}

template <typename T, int rows, int cols>
struct TileShape2D<T, rows, cols, Layout::NZ>
    : public Shape<1, GetTileShape2DNZCols<T, cols>(), GetTileShape2DNZRows<T, rows>(), FRACTAL_NZ_ROW,
                   C0_SIZE_BYTE / sizeof(T)> {
    static constexpr int C0Size = C0_SIZE_BYTE / sizeof(T);
    using Parent = Shape<1, GetTileShape2DNZCols<T, cols>(),
                         GetTileShape2DNZRows<T, rows>(), FRACTAL_NZ_ROW, C0Size>;

    static_assert((rows == DYNAMIC) || (rows % FRACTAL_NZ_ROW == 0), "rows must be divisible by 16 for Layout::NZ");
    static_assert((cols == DYNAMIC) || (cols % C0Size == 0), "cols must be divisible by C0Size for Layout::NZ");

    __aicore__ PTO_INLINE TileShape2D() : Parent() {}

    __aicore__ PTO_INLINE TileShape2D(int dynamicRows, int dynamicCols)
        : Parent(1, dynamicCols / C0Size, dynamicRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, C0Size)
    {
    }
    using Parent::Parent;
};

template <typename T, int cols>
constexpr int GetShape2DCols()
{
    if constexpr (cols == DYNAMIC) {
        return DYNAMIC;
    } else {
        return cols;
    }
}
template <typename T, int rows>
constexpr int GetShape2DRows()
{
    if constexpr (rows == DYNAMIC) {
        return DYNAMIC;
    } else {
        return rows;
    }
}
template <typename T, int rows, int cols>
struct TileShape2D<T, rows, cols, Layout::ND>
    : public Shape<1, 1, 1, GetShape2DRows<T, rows>(),
                   GetShape2DCols<T, cols>()> {
    using Parent = Shape<1, 1, 1, GetShape2DRows<T, rows>(),
                         GetShape2DCols<T, cols>()>;

    __aicore__ PTO_INLINE TileShape2D() : Parent() {}

    __aicore__ PTO_INLINE TileShape2D(int dynamicRows, int dynamicCols) : Parent(1, 1, 1, dynamicRows, dynamicCols) {}
    using Parent::Parent;
};
template <typename T, int rows, int cols>
struct TileShape2D<T, rows, cols, Layout::DN>
    : public Shape<1, 1, 1, GetShape2DRows<T, rows>(),
                   GetShape2DCols<T, cols>()> {
    using Parent = Shape<1, 1, 1, GetShape2DRows<T, rows>(),
                         GetShape2DCols<T, cols>()>;

    __aicore__ PTO_INLINE TileShape2D() : Parent() {}

    __aicore__ PTO_INLINE TileShape2D(int dynamicRows, int dynamicCols) : Parent(1, 1, 1, dynamicRows, dynamicCols) {}
    using Parent::Parent;
};

template <typename T, int rows = DYNAMIC, int cols = DYNAMIC, Layout Layout_ = Layout::ND>
struct BaseShape2D;

template <typename T, int cols>
constexpr int GetBaseShape2DNZCols()
{
    if constexpr (cols == DYNAMIC) {
        return DYNAMIC;
    } else {
        return static_cast<int>(cols / (C0_SIZE_BYTE / sizeof(T)));
    }
}

template <typename T, int rows, int cols>
constexpr int GetBaseShape2DStride0()
{
    if constexpr (cols == DYNAMIC || rows == DYNAMIC) {
        return DYNAMIC;
    } else {
        return static_cast<int>(cols * rows);
    }
}
template <typename T, int rows>
constexpr int GetBaseShape2DStride1()
{
    if constexpr (rows == DYNAMIC) {
        return DYNAMIC;
    } else {
        return static_cast<int>(rows * (C0_SIZE_BYTE / sizeof(T)));
    }
}
template <typename T, int rows, int cols>
struct BaseShape2D<T, rows, cols, Layout::NZ>
    : public Stride<GetBaseShape2DStride0<T, rows, cols>(),
                    GetBaseShape2DStride1<T, rows>(),
                    FRACTAL_NZ_ROW * (C0_SIZE_BYTE / sizeof(T)), C0_SIZE_BYTE / sizeof(T), 1> {
    static constexpr int C0Size = C0_SIZE_BYTE / sizeof(T);
    static constexpr int FractalNZSize = FRACTAL_NZ_ROW * (C0_SIZE_BYTE / sizeof(T));
    using Parent = Stride<GetBaseShape2DStride0<T, rows, cols>(),
                          GetBaseShape2DStride1<T, rows>(), FractalNZSize, C0Size, 1>;
    static_assert((rows == DYNAMIC) || (rows % FRACTAL_NZ_ROW == 0), "rows must be divisible by 16 for Layout::NZ");
    static_assert((cols == DYNAMIC) || (cols % C0Size == 0), "cols must be divisible by C0Size for Layout::NZ");

    __aicore__ PTO_INLINE BaseShape2D() : Parent() {}

    __aicore__ PTO_INLINE BaseShape2D(int dynamicRows, int dynamicCols)
        : Parent(dynamicCols * dynamicRows, dynamicRows * C0Size, FractalNZSize, C0Size, 1)
    {
    }
    using Parent::Parent;
};
template <typename T, int rows, int cols>
struct BaseShape2D<T, rows, cols, Layout::ND>
    : public Stride<GetBaseShape2DStride0<T, rows, cols>(),
                    GetBaseShape2DStride0<T, rows, cols>(),
                    GetBaseShape2DStride0<T, rows, cols>(),
                    GetShape2DCols<T, cols>(), 1> {
    using Parent = Stride<GetBaseShape2DStride0<T, rows, cols>(),
                          GetBaseShape2DStride0<T, rows, cols>(),
                          GetBaseShape2DStride0<T, rows, cols>(),
                          GetShape2DCols<T, cols>(), 1>;

    __aicore__ PTO_INLINE BaseShape2D() : Parent() {}

    __aicore__ PTO_INLINE BaseShape2D(int dynamicRows, int dynamicCols)
        : Parent(dynamicRows * dynamicCols, dynamicRows * dynamicCols, dynamicRows * dynamicCols, dynamicCols, 1)
    {
    }
    using Parent::Parent;
};
template <typename T, int rows, int cols>
struct BaseShape2D<T, rows, cols, Layout::DN>
    : public Stride<GetBaseShape2DStride0<T, rows, cols>(),
                    GetBaseShape2DStride0<T, rows, cols>(),
                    GetBaseShape2DStride0<T, rows, cols>(), 1,
                    GetShape2DRows<T, rows>()> {
    using Parent = Stride<GetBaseShape2DStride0<T, rows, cols>(),
                          GetBaseShape2DStride0<T, rows, cols>(),
                          GetBaseShape2DStride0<T, rows, cols>(), 1,
                          GetShape2DRows<T, rows>()>;

    __aicore__ PTO_INLINE BaseShape2D() : Parent() {}

    __aicore__ PTO_INLINE BaseShape2D(int dynamicRows, int dynamicCols)
        : Parent(dynamicRows * dynamicCols, dynamicRows * dynamicCols, dynamicRows * dynamicCols, 1, dynamicRows)
    {
    }
    using Parent::Parent;
};

template <typename TileData>
__aicore__ void TASSIGN_IMPL(TileData &tile, uint32_t addr);

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout BFractal_ = BLayout::RowMajor,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_,
          const SLayout SFractal_ = SLayout::NoneBox,
          const int SFractalSize_ = 512,
          const PadValue PadVal_ = PadValue::Null>
struct Tile {
  public:
    using DType = Element_;

    static constexpr int getInnerRow() {
        if constexpr (SFractalSize_ == 1024) {
            static_assert(sizeof(DType) == 4, "Size of datatype != 4");
            return 16;
        } else {
            return isBoxedLayout
                     ? (isInnerRowMajor ? 16 : byteSize / sizeof(DType))
                     : 1;
        }
    }

    static constexpr int getInnerCol() {
        if constexpr (SFractalSize_ == 1024) {
            static_assert(sizeof(DType) == 4, "Size of datatype != 4");
            return 16;
        } else {
            return isBoxedLayout
                     ? (isInnerRowMajor ? byteSize / sizeof(DType) : 16)
                     : 1;
        }
    }

    static constexpr Location Loc = Loc_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int RowStride = BFractal_ == BLayout::RowMajor ? Cols : 1;
    static constexpr int ColStride = BFractal_ == BLayout::RowMajor ? 1 : Rows;

    static constexpr int ValidRow = RowValid_;
    static constexpr int ValidCol = ColValid_;
    static_assert(Rows > 0 && ValidRow <= Rows && Cols > 0 && ValidCol <= Cols,
                  "Invalid Tile Layout.");

    static constexpr SLayout SFractal = SFractal_;
    static constexpr int Numel = Rows * Cols;
    static constexpr bool isRowMajor = BFractal_ == BLayout::RowMajor;

    static constexpr int SFractalSize = SFractalSize_;
    static constexpr PadValue PadVal = PadVal_;

    // constructor for static shape
    __aicore__ Tile(){};
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(std::enable_if_t<(RowMask > 0) && (ColMask > 0), DType> val);

    // constructor for both dimensions are runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
         std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(DType val, std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
         std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);

    // constructor for row dimension is runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(DType val,
         std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);

    // constructor for col dimension is runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    __aicore__
    Tile(DType val,
         std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);

    static constexpr int byteSize = 32;
    static constexpr bool isBoxedLayout = (SFractal != SLayout::NoneBox);
    static constexpr bool isInnerRowMajor = (SFractal == SLayout::RowMajor);
    static constexpr bool isInnerColMajor = (SFractal == SLayout::ColMajor);

    static constexpr int InnerRows = getInnerRow();
    static constexpr int InnerCols = getInnerCol();

    static constexpr int InnerNumel = InnerRows * InnerCols;

    static_assert(Rows % InnerRows == 0,
                  "Layout rows must be divisible by inner box rows");
    static_assert(Cols % InnerCols == 0,
                  "Layout cols must be divisible by inner box cols");

    static_assert(
        (BFractal_ == BLayout::RowMajor && SFractal_ == SLayout::NoneBox && Cols * sizeof(DType) % 32 == 0) ||
        (BFractal_ == BLayout::ColMajor && SFractal_ == SLayout::NoneBox && Rows * sizeof(DType) % 32 == 0) ||
        (SFractal_ != SLayout::NoneBox) && (Rows % InnerRows == 0 && Cols % InnerCols == 0),
        "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align, \
         BFractal_ is ColMajor and SFractal_ is NoneBox: Cols must be 32 bytes align, \
         SFractal_ in not NoneBox: Rows/Cols must be integer multiple of InnerRows/InnerCols."
         );

    static_assert(SFractalSize_ == 512 || SFractalSize_ == 1024,
                  "SFractalSize_ illegal");

#ifdef __CPU_SIM
    using TileDType = Tile::DType[Rows*Cols];
#else
    #ifdef __PTO_AUTO__
        using TileDType = typename MemoryQualifier<Loc, DType>::type tile_size(Rows * Cols);
    #else
        using TileDType = typename MemoryQualifier<Loc, DType>::type;
    #endif
#endif

    __aicore__ TileDType &data() { return data_; }
    __aicore__ const TileDType &data() const { return data_; }

    int RowMaskInternal;
    int ColMaskInternal;

    template <int RowMask = ValidRow>
    __aicore__ static constexpr std::enable_if_t<(RowMask > 0), int> GetValidRow() {
        return RowMask;
    }

    template <int RowMask = ValidRow>
    __aicore__ std::enable_if_t<RowMask == -1, int> GetValidRow() const {
        return RowMaskInternal;
    }

    template <int ColMask = ValidCol>
    __aicore__ static constexpr std::enable_if_t<(ColMask > 0), int> GetValidCol() {
        return ColMask;
    }

    template <int ColMask = ValidCol>
    __aicore__ std::enable_if_t<ColMask == -1, int> GetValidCol() const {
        return ColMaskInternal;
    }

    friend __aicore__ void
    TASSIGN_IMPL<>(Tile<Loc_, Element_, Rows_, Cols_, BFractal_, RowValid_,
                   ColValid_, SFractal_, SFractalSize_, PadVal_> &tile,
              uint32_t addr);

  private:
    __aicore__ void assignData(TileDType data) { data_ = data; }
    TileDType data_;
};

#ifdef __DAV_V220
template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileLeft = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::RowMajor,
                      RowValid_, ColValid_, SLayout::RowMajor, 512>;
#endif

#if defined (__DAV_V310) || defined (__CPU_SIM)
template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileLeft = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::ColMajor,
                      RowValid_, ColValid_, SLayout::RowMajor, 512>;
#endif


template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileRight =
  Tile<Location::Right, Element_, Rows_, Cols_, BLayout::RowMajor, RowValid_,
       ColValid_, SLayout::ColMajor, 512>;

template <typename Element_, const int Rows_, const int Cols_,
          const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileAcc = Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::ColMajor,
                     RowValid_, ColValid_, SLayout::RowMajor, 1024>;

template <typename T> struct is_global : std::false_type {};
template <typename T> struct is_tile : std::false_type {
    static constexpr SLayout layout_enum = SLayout::NoneBox;
};

template <typename Element_, typename Layout_, typename Stride_>
struct is_global<GlobalTensor<Element_, Layout_, Stride_>> : std::true_type {};

template <Location Loc_, typename Element_, const int Rows_, const int Cols_,
          const BLayout BFractal_, const int RowValid_, const int ColValid_,
          const SLayout SFractal_, const int SFractalSize_>
struct is_tile<Tile<Loc_, Element_, Rows_, Cols_, BFractal_, RowValid_,
                    ColValid_, SFractal_, SFractalSize_>> : std::true_type {
    static constexpr SLayout layout_enum = SFractal_;
};

template <typename T>
constexpr bool is_boxed_tile =
    is_tile<T>::value && (is_tile<T>::layout_enum != SLayout::NoneBox);

template <typename tile_shape> struct is_Nz_layout {
  static constexpr bool value = !tile_shape::isRowMajor &&
                                tile_shape::isBoxedLayout &&
                                tile_shape::isInnerRowMajor;
};

template <typename tile_shape> struct is_Zn_layout {
  static constexpr bool value = tile_shape::isRowMajor &&
                                tile_shape::isBoxedLayout &&
                                tile_shape::isInnerColMajor;
};

template <typename T> constexpr bool is_global_data_v = is_global<T>::value;

template <typename T> constexpr bool is_tile_data_v = is_tile<T>::value;

template <typename T> constexpr bool is_boxed_data_v = is_boxed_tile<T>;

} // namespace pto

#endif
