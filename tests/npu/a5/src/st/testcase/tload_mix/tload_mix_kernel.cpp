/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_ND2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::ND>;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>,
        Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;  // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    // L1清0 方便测试非对齐场景
#if defined(__DAV_CUBE__)
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    if constexpr (std::is_same<T, float4_e1m2x2_t>::value ||
                  std::is_same<T, float4_e2m1x2_t>::value) {
        blockLen = baseM * baseK / 64;
    }
    int64_t repeatBit =
        (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    create_cbuf_matrix((__cbuf__ uint16_t *)srcMatAddr, repeatBit, 0);
#endif

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}
template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_DN2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>,
        Layout::DN>;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>,
        Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;  // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    // L1清0 方便测试非对齐场景
#if defined(__DAV_CUBE__)
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    int64_t repeatBit =
        (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    create_cbuf_matrix((__cbuf__ uint16_t *)srcMatAddr, repeatBit, 0);
#endif

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_ND2ND(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::ND>;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>,
        Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::NoneBox>;  // 大N小Z
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    if constexpr (std::is_same<T, float4_e1m2x2_t>::value ||
                  std::is_same<T, float4_e2m1x2_t>::value) {
        blockLen = baseM * baseK / 64;
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_DN2DN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>,
        Layout::DN>;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseK, baseM>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseM, 1>,
        Layout::ND>;  // actually is DN

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, M, K, SLayout::NoneBox>;  // 大N小Z
    using TileUBData =
        Tile<TileType::Vec, T, baseK, baseM, BLayout::RowMajor, -1, -1>;  // DN：baseM need 32Byte aligned
    TileUBData srcTile(baseK, baseM);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_NZ2NZ(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, WN5, 1>,
        Layout::NZ>;  // [2,2,4,16,8]
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>,
        Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);
    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::ColMajor, N3 * M, N2 * K, SLayout::RowMajor, 512>;  // [80,48]
                                                                                                          // valid is
                                                                                                          // [N3 * M, N2
                                                                                                          // * K]

    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    // L1清0 方便测试非对齐场景
#if defined(__DAV_CUBE__)
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    int64_t repeatBit =
        (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    create_cbuf_matrix((__cbuf__ uint16_t *)srcMatAddr, repeatBit, 0);
#endif

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}


template <typename T, int N1, int N2, int N3, int M, int K, int WN1, int WN2, int WN3, int WN4, int WN5, int baseM,
    int baseK>
AICORE inline void runTLOAD_MIX_DN2ZN(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<T,
        pto::Shape<N1, N2, N3, M, K>,
        pto::Stride<WN2 * WN3 * WN4 * WN5, WN3 * WN4 * WN5, WN4 * WN5, 1, WN4>,
        Layout::DN>;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>,
        Layout::ND>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>;  // 大Z小N
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    // L1清0 方便测试非对齐场景
#if defined(__DAV_CUBE__)
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
    if constexpr (std::is_same<T, float4_e1m2x2_t>::value ||
                  std::is_same<T, float4_e2m1x2_t>::value) {
        blockLen = baseM * baseK / 64;
    }
    int64_t repeatBit =
        (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | 1;
    create_cbuf_matrix((__cbuf__ uint16_t *)srcMatAddr, repeatBit, 0);
#endif
    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0, 0);  // move to vector
                                                                                                  // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(
        PIPE_MTE3, syncID);  // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (format == 0) {  // ND2NZ
        runTLOAD_MIX_ND2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 1) {  // DN2NZ
        runTLOAD_MIX_DN2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 2) {  // ND2ND
        runTLOAD_MIX_ND2ND<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 3) {  // DN2DN
        runTLOAD_MIX_DN2DN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 4) {  // NZ2NZ
        runTLOAD_MIX_NZ2NZ<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    } else if constexpr (format == 5) {  // DN2ZN
        runTLOAD_MIX_DN2ZN<T, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
            reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), reinterpret_cast<__gm__ T *>(src1));
    }
}

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADMIX(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    TLOAD_MIX_KERNEL<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

/********************format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ*****************************/
// 2:ND2ND
template void launchTLOADMIX<int8_t, 2, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 2, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// 0:ND2NZ
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// 1:DN2NZ
template void launchTLOADMIX<uint16_t, 1, 1, 1, 1, 64, 128, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<float, 1, 1, 1, 1, 51, 123, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 1, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// 3:DN2DN
template void launchTLOADMIX<float, 3, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 3, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 3, 1, 2, 3, 64, 128, 1, 3, 4, 96, 128, 64, 768>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// 4.NZ2NZ
template void launchTLOADMIX<float, 4, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void launchTLOADMIX<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<uint64_t, 2, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// 5 DN2ZN
template void launchTLOADMIX<uint16_t, 5, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX<int8_t, 5, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);


template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
    int WN5, int BASEM, int BASEK>
__global__ AICORE void TLOAD_MIX_KERNEL_B4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1) {
    if constexpr (format == 0) { // ND2NZ
        if constexpr (dtype == 0) {
            runTLOAD_MIX_ND2NZ<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_ND2NZ<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 1) { // DN2NZ
        static_assert(format != 1, "DN2NZ not support if input dtype is fp4");
    } else if constexpr (format == 2) { // ND2ND
        if constexpr (dtype == 0) {
            runTLOAD_MIX_ND2ND<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_ND2ND<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 3) { // DN2DN
        if constexpr (dtype == 0) {
            runTLOAD_MIX_DN2DN<float4_e1m2x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_DN2DN<float4_e2m1x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 4) { // NZ2NZ
        if constexpr (dtype == 0) {
            runTLOAD_MIX_NZ2NZ<float4_e1m2x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_NZ2NZ<float4_e2m1x2_t, N1, N2, N3, N4, N5 * 2, WN1, WN2, WN3, WN4, WN5 * 2, BASEM, BASEK * 2>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    } else if constexpr (format == 5) { // DN2ZN
        if constexpr (dtype == 0) {
            runTLOAD_MIX_DN2ZN<float4_e1m2x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(out), reinterpret_cast<__gm__ float4_e1m2x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e1m2x2_t *>(src1));
        } else if constexpr (dtype == 1) {
            runTLOAD_MIX_DN2ZN<float4_e2m1x2_t, N1, N2, N3, N4 * 2, N5, WN1, WN2, WN3, WN4 * 2, WN5, BASEM * 2, BASEK>(
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(out), reinterpret_cast<__gm__ float4_e2m1x2_t *>(src0),
                reinterpret_cast<__gm__ float4_e2m1x2_t *>(src1));
        }
    }
}

template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
    int WN5, int BASEM, int BASEK>
void launchTLOADMIX_B4(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    TLOAD_MIX_KERNEL_B4<T, format, dtype, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

template void launchTLOADMIX_B4<uint8_t, 2, 0, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 2, 1, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 0, 1, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 4, 0, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 3, 0, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTLOADMIX_B4<uint8_t, 5, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);