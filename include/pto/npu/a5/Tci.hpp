/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCI_HPP
#define TCI_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <typename TileData, typename T>
__aicore__
PTO_INLINE
void CheckValid() {
    static_assert((std::is_same<typename TileData::DType, T>::value),
    "expect src and dst same datatype");
    static_assert((sizeof(typename TileData::DType) == 4 || (sizeof(typename TileData::DType) == 2)),
    "expect b32 or b16");
    static_assert((TileData::Cols != 1),
    "expect row is 1");
}

template <typename TileData, typename T, int descending = 0>
__tf__
__aicore__
void Tci(typename TileData::TileDType __out__ dst, T S, unsigned validCol)
{
    // 1.获取dst中的信息;
    using Tdst = typename TileData::DType;
    __ubuf__ Tdst *dstPtr = (__ubuf__ Tdst *)__cce_get_tile_ptr(dst);
    //scalar
    if(descending)
    {
        for(int32_t j = 0; j < validCol; j++) {
            *(dstPtr + j) = S - j;
        }
    }
    else
    {
        for(int32_t j = 0; j < validCol; j++) {
            *(dstPtr + j) = S + j;
        }
    }
}

template <typename TileData, typename T, int descending>
__aicore__
void TCI_IMPL (TileData &dst, T S)
{
    CheckValid<TileData, T>();
    unsigned validCol = dst.GetValidCol();
    Tci<TileData, T, descending>(dst.data(), S, validCol);
}
}
#endif