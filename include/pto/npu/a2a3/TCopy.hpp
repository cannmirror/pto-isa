/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOPY_HPP
#define TCOPY_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto{
    template <typename T> 
    PTO_INTERNAL void CopyCountMode(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol){
        
        set_mask_count();
        SetVectorCount(validCol*validRow);
        uint64_t blockLen = (validCol*validRow*sizeof(T) + BLOCK_BYTE_SIZE-1) / BLOCK_BYTE_SIZE;
        copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, blockLen, 1, 1);
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }

    template <typename T, unsigned dstStride, unsigned srcStride>
    PTO_INTERNAL void Copy2LCountMode(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol){
        
        set_mask_count();
        SetVectorCount(validCol);
        uint64_t blockLen = (validCol*sizeof(T) + BLOCK_BYTE_SIZE-1) / BLOCK_BYTE_SIZE;
        for (unsigned row = 0; row < validRow; ++row) {
            copy_ubuf_to_ubuf(dstPtr + row * dstStride, srcPtr + row * srcStride, 0, 1, blockLen, 1, 1);
        }
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }

    template <typename T>
    PTO_INTERNAL void intrinsicCopy(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned repeat, unsigned offset){
        if constexpr (sizeof(T) < 4){
            vcopy((__ubuf__ uint16_t *)(dstPtr + offset), (__ubuf__ uint16_t *)(srcPtr+offset), repeat, 1, 1, 8, 8);
        } else{
            vcopy((__ubuf__ uint32_t *)(dstPtr + offset), (__ubuf__ uint32_t *)(srcPtr+offset), repeat, 1, 1, 8, 8);
        }
    }

    template <typename T, unsigned elementsPerRepeat>
    PTO_INTERNAL void CopyNormMode(__ubuf__ T*dstPtr, __ubuf__ T* srcPtr, unsigned validRow, unsigned validCol){
        unsigned numElements = validRow * validCol;
        unsigned headRepeats = numElements / elementsPerRepeat;
        unsigned tailElements = numElements % elementsPerRepeat;
        intrinsicCopy<T>(dstPtr, srcPtr, headRepeats, 0);
        if (tailElements > 0){
            unsigned offset = headRepeats * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            intrinsicCopy<T>(dstPtr, srcPtr, 1, offset);
            SetFullVecMaskByDType<T>();
        }
    }
    
    template<typename TileDataDst, typename TileDataSrc, unsigned blockSizeElem, unsigned srcStride, unsigned dstStride>
    __tf__ PTO_INTERNAL void TCopy(typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src, uint64_t validRow, uint64_t validCol) {
        if (validRow ==0 || validCol == 0) {
            return;
        }
        using T = typename TileDataSrc::DType;
        using U = typename TileDataDst::DType;
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

        static_assert(sizeof(T) == sizeof(U), "TMOV: src and dst data type is different!");
        
        if constexpr ((TileDataDst::Cols == TileDataDst::ValidCol && TileDataSrc::Cols == TileDataSrc::ValidCol) || (TileDataDst::Rows == 1)){
            constexpr unsigned totalRepeats = (TileDataDst::Cols * TileDataDst::Rows + elementsPerRepeat - 1) / elementsPerRepeat;
            if constexpr (totalRepeats > REPEAT_MAX) {
                CopyCountMode<U>(dstPtr, srcPtr, validRow, validCol);
            } else {
                CopyNormMode<T, elementsPerRepeat>(dstPtr, srcPtr, validRow, validCol);
            }
        } else {
            if (TileDataDst::Cols == validCol || validRow == 1){
                unsigned totalRepeats = (validRow*validCol + elementsPerRepeat - 1) / elementsPerRepeat;
                if(totalRepeats > REPEAT_MAX){
                    CopyCountMode<U>(dstPtr, srcPtr, validRow, validCol);
                } else {
                    CopyNormMode<T, elementsPerRepeat>(dstPtr, srcPtr, validRow, validCol);
                }
            } else {
                Copy2LCountMode<U, dstStride, srcStride>(dstPtr, srcPtr, validRow, validCol);
            }
        }
    }  // end of tf
}
#endif