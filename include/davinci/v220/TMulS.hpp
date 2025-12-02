#ifndef TMULS_HPP
#define TMULS_HPP

#include <functional>
#include "common/constants.hpp"

namespace pto
{
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __aicore__ void TMulS(typename TileData::TileDType __out__ dst,
                                 typename TileData::TileDType __in__ src0,
                                 typename TileData::DType __in__ src1,
                                 unsigned numRepeatPerLine,
                                 unsigned numRemainPerLine,
                                 unsigned validRow)
    {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        if (numRepeatPerLine > 0)
        {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++)
            {
                if (numLoop > 0)
                {
                    for (int j = 0; j < numLoop; j++)
                    {
                        vmuls(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                              src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                              src1,
                              REPEAT_MAX, 1, 1, 8, 8);
                    }
                }
                if (remainAfterLoop > 0)
                {
                    vmuls(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                          src0Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                          src1,
                          remainAfterLoop, 1, 1, 8, 8);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine > 0)
        {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop > 0)
            {
                for (int i = 0; i < numLoop; i++)
                {
                    if (strideOverFlag)
                    {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++)
                        {
                            vmuls(dstPtr + i * REPEAT_MAX * stride + j * stride,
                                  src0Ptr + i * REPEAT_MAX * stride + j * stride,
                                  src1,
                                  1, 1, 1, 1, 1);
                        }
                    }
                    else
                    {
                        vmuls(dstPtr + i * REPEAT_MAX * stride,
                              src0Ptr + i * REPEAT_MAX * stride,
                              src1,
                              REPEAT_MAX, 1, 1, static_cast<uint8_t>(stride / blockSizeElem), static_cast<uint8_t>(stride / blockSizeElem));
                    }
                }
            }
            if (remainAfterLoop > 0)
            {
                if (strideOverFlag)
                {
                    for (unsigned j = 0; j < remainAfterLoop; j++)
                    {
                        vmuls(dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                              src0Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                              src1,
                              1, 1, 1, 1, 1);
                    }
                }
                else
                {
                    vmuls(dstPtr + numLoop * REPEAT_MAX * stride,
                          src0Ptr + numLoop * REPEAT_MAX * stride,
                          src1,
                          remainAfterLoop, 1, 1, static_cast<uint8_t>(stride / blockSizeElem), static_cast<uint8_t>(stride / blockSizeElem));
                }
            }
            set_vector_mask(-1, -1);
        }
    }
    template <typename TileData>
    __aicore__ void TMULS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        TMulS<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), scalar,
                                                                  numRepeatPerLine, numRemainPerLine, validRow);
    }
}

#endif