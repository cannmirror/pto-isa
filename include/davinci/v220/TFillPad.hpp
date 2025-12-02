/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFILLPAD_HPP
#define TFILLPAD_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "TLoad.hpp"

namespace pto {
template <typename TileData>
__aicore__ constexpr auto getCopyNullPtr()
{
    using T = typename TileData::DType;
    if constexpr (sizeof(T) == 4) {
        return (__ubuf__ uint32_t*) 0;
    }
    else if constexpr (sizeof(T) == 2) {
        return (__ubuf__ uint16_t*) 0;
    }
    else if constexpr (sizeof(T) == 1) {
        return (__ubuf__ uint16_t*) 0;
    }
    else
    {
        static_assert(sizeof(T) < 0, "TFILLPAD: Unsupported DType for PadValue!");
    }
}

template <typename TileDataDst, typename TileDataSrc>
__tf__
__aicore__
PTO_INLINE
void TFillPad_CopyData(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src, uint64_t dstValidRow, uint64_t dstValidCol, uint64_t srcValidRow, uint64_t srcValidCol)
{
    set_mask_count(); // counter mode
    using T = typename TileDataSrc::DType;
    auto srcPtr = getCopyNullPtr<TileDataSrc>();
    auto dstPtr = getCopyNullPtr<TileDataDst>();
    srcPtr = (decltype(srcPtr)) __cce_get_tile_ptr(src);
    dstPtr = (decltype(dstPtr)) __cce_get_tile_ptr(dst);
    constexpr uint64_t copySrcCols = (sizeof(T) == 1) ? TileDataSrc::Cols/2 : TileDataSrc:: Cols;
    constexpr uint64_t copyDstCols = (sizeof(T) == 1) ? TileDataDst::Cols/2 : TileDataDst:: Cols;

    set_vector_mask(0, copySrcCols);

    uint64_t srcCopyRow = srcValidRow;
    auto _srcPtr = srcPtr;
    auto _dstPtr = dstPtr;
    if constexpr (TileDataSrc::Rows > 255)
    {
        while (srcCopyRow > 255)
        {
            uint8_t repeat = 255;
            uint16_t srcRepeatStride = TileDataSrc:: Cols * sizeof(T) / 32;
            uint16_t dstRepeatStride = TileDataDst:: Cols * sizeof(T) / 32;
            vcopy(_dstPtr,
                  _srcPtr,
                  repeat,
                  1,
                  1,
                  dstRepeatStride,
                  srcRepeatStride);
            srcCopyRow -= 255;
            _srcPtr += 255 * copySrcCols;
            _dstPtr += 255 * copyDstCols;
        }
    }
    uint8_t repeat = srcCopyRow;
    uint16_t srcRepeatStride = TileDataSrc:: Cols * sizeof(T) / 32;
    uint16_t dstRepeatStride = TileDataDst:: Cols * sizeof(T) / 32;
    vcopy(_dstPtr,
          _srcPtr,
          repeat,
          1,
          1,
          dstRepeatStride,
          srcRepeatStride);
}

template <typename T>
__aicore__
PTO_INLINE
uint64_t getPadMask(uint64_t validCol)
{
    if constexpr (sizeof(T) == 4) {
        return 0;
    }
    else if constexpr (sizeof(T) == 2) {
        return 0;
    }
    else if constexpr (sizeof(T) == 1) {
        return 0;
    }
    else
    {
        static_assert(sizeof(T) < 0, "TFILLPAD: Unsupported DType for PadValue!");
    }
}

template <typename TileDataDst, typename TileDataSrc>
__tf__
__aicore__
PTO_INLINE
void TFillPad(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src, uint64_t dstValidRow, uint64_t dstValidCol, uint64_t srcValidRow, uint64_t srcValidCol)
{
    using T = typename TileDataSrc::DType;
    auto srcPtr = getCopyNullPtr<TileDataSrc>();
    auto dstPtr = getCopyNullPtr<TileDataDst>();
    srcPtr = (decltype(srcPtr)) __cce_get_tile_ptr(src);
    dstPtr = (decltype(dstPtr)) __cce_get_tile_ptr(dst);
    auto padValue = getPadValue<TileDataDst>();

    constexpr const uint64_t copyDstCols = sizeof(T) == 1 ? TileDataDst::Cols / 2 : TileDataDst::Cols;
    uint64_t elements_per_block = (sizeof(T) == 1) ? 16 : 32 / sizeof(T);
    uint64_t srcValidCol32B = (sizeof(T) == 1) ? CeilDivision(CeilDivision(srcValidCol, 2), elements_per_block) * elements_per_block : 
                            CeilDivision(srcValidCol, elements_per_block) * elements_per_block;
    uint64_t padOffset = srcValidCol32B;
    uint64_t padCols = copyDstCols - srcValidCol32B;

    if constexpr (TileDataDst::PadVal != TileDataSrc::PadVal)
    {
        if constexpr (sizeof(T) == 1)
        {
            uint64_t pad_32B = 32 / sizeof(T) - srcValidCol;
            set_flag(PIPE_V, PIPE_S, (event_t)0);
            wait_flag(PIPE_V, PIPE_S, (event_t)0);
            using TP = decltype(padValue);
            for (uint64_t r = 0; r < srcValidRow; r++)
            {
                __ubuf__ TP* dstPadPtr = &((__ubuf__ TP*) dstPtr)[r * TileDataDst::Cols + srcValidCol];
                for (uint64_t p = 0; p < pad_32B; p++)
                {
                    *(dstPadPtr++) = padValue;
                }
            }
            dsb(DSB_UB);
        }
        else
        {
            uint64_t pad_32B = srcValidCol32B - srcValidCol;
            set_mask_norm();
            uint64_t mask = 0;
            uint16_t dstRepeatStride = TileDataDst::Cols * sizeof(T) / 32;
            if constexpr (sizeof(T) == 4)
                mask = 0xffULL; //all elements;
            else
                mask = 0xffffULL; //all elements;
            mask = mask >> (elements_per_block - pad_32B);
            mask = mask << (elements_per_block - pad_32B);
            set_vector_mask(0, mask);
            uint64_t fillRow = srcValidRow;
            auto _dstPtr = dstPtr + (srcValidCol32B - elements_per_block);
            if constexpr (TileDataSrc::Rows > 255)
            {
                vector_dup(_dstPtr, padValue, 255, 1, 1, dstRepeatStride, 0);
                _dstPtr += 255 * TileDataDst::Cols;
                fillRow -= 255;
            }
            vector_dup(_dstPtr, padValue, fillRow, 1, 1, dstRepeatStride, 0);
            pipe_barrier(PIPE_V);
        }
    }

    uint64_t dupPadValue = sizeof(T) == 1 ? ((uint64_t)padValue) << 8 | ((uint64_t)padValue) : padValue;

    set_mask_count();  // counter mode
    set_vector_mask(0, padCols);
    vector_dup(dstPtr + padOffset, dupPadValue, 1, 1, 1, 8, 0);  // pad single row

    pipe_barrier(PIPE_V);

    if constexpr (TileDataSrc::Rows > 1)
    {
        auto _dstPtr = dstPtr + padOffset + copyDstCols;
        uint64_t fillRow = srcValidRow - 1;
        uint16_t dstRepeatStride = TileDataDst::Cols * sizeof(T) / 32;
        if constexpr (TileDataSrc::Rows > 256)
        {
            while (fillRow > 255)
            {
                uint8_t repeat = 255;
                vcopy(_dstPtr,
                      dstPtr + padOffset,
                      repeat,
                      1,
                      0,
                      dstRepeatStride,
                      0);
                _dstPtr += 255 * copyDstCols;
                fillRow -= 255;
            }
        }
        uint8_t repeat = fillRow;
        vcopy(_dstPtr,
              dstPtr + padOffset,
              repeat,
              1,
              0,
              dstRepeatStride,
              0);
    }

    int padRows = dstValidRow - srcValidRow;
    set_vector_mask(0, padRows * copyDstCols);
    vector_dup(dstPtr + srcValidRow * copyDstCols, dupPadValue, 1, 1, 1, 8, 0); //pad 2d->1d rows

    set_mask_norm();  // restore to norm mode
    set_vector_mask(-1, -1);
}  //end of tf

template <typename TileDataDst, typename TileDataSrc, bool inplace>
__aicore__
PTO_INLINE
void TFILLPAD_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validDstCol = dst.GetValidCol();

    using T = typename TileDataSrc::DType;
    using U = typename TileDataDst::DType;
    static_assert(TileDataDst::PadVal != PadValue::Null, "TFillPad: dst vecTile pad value must not be Null!");
    static_assert(sizeof(T) == sizeof(U), "TFillPad: src and dst data type is different!");
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TFillPad: Invalid data type.");

    if (validDstRow == 0 || validDstCol == 0) {
        return;
    }
    if constexpr (!inplace)
    {
        TFillPad_CopyData<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
    }
    TFillPad<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
}

template <typename TileDataDst, typename TileDataSrc>
__aicore__
PTO_INLINE
void TFILLPAD(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
        "TFillPad: Dst/Src vecTile Rows/Cols must be the same.");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
__aicore__
PTO_INLINE
void TFILLPAD_INPLACE(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
        "TFillPad: Dst vecTile Rows/Cols must be greater or equal to src vecTile.");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, true>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
__aicore__
PTO_INLINE
void TFILLPAD_EXPAND(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols >= TileDataSrc::Cols && TileDataDst::Rows >= TileDataSrc::Rows,
        "TFillPad: Dst/Src vecTile Rows/Cols must be the same.");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
}

}  // namespace pto
#endif