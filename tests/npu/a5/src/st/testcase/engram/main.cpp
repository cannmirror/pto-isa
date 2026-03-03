/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

static std::string GetGoldenDir()
{
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    return std::string("../") + info->test_suite_name() + "." + info->name();
}

template <typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
void LaunchEngramBaseline(float *out, float *table, TIdx *indices, float *hid, float *gw, void *stream);

template <typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
void LaunchEngramFused(float *out, float *table, TIdx *indices, float *hid, float *gw, void *stream);

enum class Variant
{
    Baseline,
    Fused
};

template <int kTableRows, int kTableCols, int kNumHeads, int kEmbDim, Variant V>
void test_engram()
{
    using TIdx = int32_t;

    constexpr size_t tableElems = (size_t)kTableRows * kTableCols;
    constexpr size_t outElems = (size_t)kEmbDim;

    const size_t tableBytes = tableElems * sizeof(float);
    const size_t idxBytes = (size_t)kNumHeads * sizeof(TIdx);
    const size_t hidBytes = (size_t)kEmbDim * sizeof(float);
    const size_t gwBytes = (size_t)kEmbDim * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);

    string gDir = GetGoldenDir();

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *hTable, *hHidden, *hGateW, *hOutput;
    TIdx *hIdx;
    aclrtMallocHost((void **)&hTable, tableBytes);
    aclrtMallocHost((void **)&hIdx, idxBytes);
    aclrtMallocHost((void **)&hHidden, hidBytes);
    aclrtMallocHost((void **)&hGateW, gwBytes);
    aclrtMallocHost((void **)&hOutput, outBytes);

    float *dTable, *dHidden, *dGateW, *dOutput;
    TIdx *dIdx;
    aclrtMalloc((void **)&dTable, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dIdx, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dHidden, hidBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dGateW, gwBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dOutput, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fSz = 0;
    ReadFile(gDir + "/table.bin", fSz, hTable, tableBytes);
    ReadFile(gDir + "/indices.bin", fSz, hIdx, idxBytes);
    ReadFile(gDir + "/hidden.bin", fSz, hHidden, hidBytes);
    ReadFile(gDir + "/gate_weight.bin", fSz, hGateW, gwBytes);

    aclrtMemcpy(dTable, tableBytes, hTable, tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dIdx, idxBytes, hIdx, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dHidden, hidBytes, hHidden, hidBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dGateW, gwBytes, hGateW, gwBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (V == Variant::Baseline) {
        LaunchEngramBaseline<TIdx, kTableRows, kTableCols, kNumHeads, kEmbDim>(dOutput, dTable, dIdx, dHidden, dGateW,
                                                                               stream);
    } else {
        LaunchEngramFused<TIdx, kTableRows, kTableCols, kNumHeads, kEmbDim>(dOutput, dTable, dIdx, dHidden, dGateW,
                                                                            stream);
    }
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hOutput, outBytes, dOutput, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(gDir + "/output.bin", hOutput, outBytes);

    vector<float> golden(outElems);
    vector<float> actual(outElems);
    ReadFile(gDir + "/golden.bin", fSz, golden.data(), outBytes);
    memcpy(actual.data(), hOutput, outBytes);

    bool pass = ResultCmp<float>(golden, actual, 0.001f);
    EXPECT_TRUE(pass);

    aclrtFree(dTable);
    aclrtFree(dIdx);
    aclrtFree(dHidden);
    aclrtFree(dGateW);
    aclrtFree(dOutput);
    aclrtFreeHost(hTable);
    aclrtFreeHost(hIdx);
    aclrtFreeHost(hHidden);
    aclrtFreeHost(hGateW);
    aclrtFreeHost(hOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

class ENGRAMTest : public ::testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(ENGRAMTest, baseline_float_128x128_8x128)
{
    test_engram<128, 128, 8, 128, Variant::Baseline>();
}

TEST_F(ENGRAMTest, baseline_float_256x256_8x256)
{
    test_engram<256, 256, 8, 256, Variant::Baseline>();
}

TEST_F(ENGRAMTest, baseline_float_512x512_8x512)
{
    test_engram<512, 512, 8, 512, Variant::Baseline>();
}

TEST_F(ENGRAMTest, baseline_float_1024x1024_8x1024)
{
    test_engram<1024, 1024, 8, 1024, Variant::Baseline>();
}

TEST_F(ENGRAMTest, fused_float_128x128_8x128)
{
    test_engram<128, 128, 8, 128, Variant::Fused>();
}

TEST_F(ENGRAMTest, fused_float_256x256_8x256)
{
    test_engram<256, 256, 8, 256, Variant::Fused>();
}

TEST_F(ENGRAMTest, fused_float_512x512_8x512)
{
    test_engram<512, 512, 8, 512, Variant::Fused>();
}

TEST_F(ENGRAMTest, fused_float_1024x1024_8x1024)
{
    test_engram<1024, 1024, 8, 1024, Variant::Fused>();
}
