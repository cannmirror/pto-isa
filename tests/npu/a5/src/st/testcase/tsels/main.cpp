/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class TSELSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream);
template <typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
          int vRows, int vCols>
void LaunchTSelsHalf(aclFloat16 *out, TMask *mask, aclFloat16 *src, aclFloat16 scalar, void *stream);

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols, bool isHalf = false>
void test_tsels()
{
    size_t dstFileSize = sizeof(T) * dstTileH * dstTileW;
    size_t maskFileSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t srcFileSize = sizeof(T) * srcTileH * srcTileW;
    size_t scalarFileSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost, *dstDevice, *srcDevice, scalar;
    TMask *maskHost, *maskDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&maskHost), maskFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/mask.bin", maskFileSize, maskHost, maskFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", scalarFileSize, &scalar, scalarFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isHalf) {
        LaunchTSelsHalf<TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>(
            dstDevice, maskDevice, srcDevice, scalar, stream);
    } else {
        LaunchTSels<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>(
            dstDevice, maskDevice, srcDevice, scalar, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(maskDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(maskHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSELSTest, case_uint8_uint8_2x32_2x32_2x32_2x32)
{
    test_tsels<uint8_t, uint8_t, 2, 32, 2, 32, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint8_uint16_2x32_2x16_2x32_2x32)
{
    test_tsels<uint8_t, uint16_t, 2, 32, 2, 16, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint8_uint32_2x32_2x8_2x32_2x32)
{
    test_tsels<uint8_t, uint32_t, 2, 32, 2, 8, 2, 32, 2, 32>();
}
TEST_F(TSELSTest, case_uint16_uint8_2x16_2x32_2x16_2x16)
{
    test_tsels<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint16_uint16_2x16_2x16_2x16_2x16)
{
    test_tsels<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint16_uint32_2x16_2x8_2x16_2x16)
{
    test_tsels<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>();
}
TEST_F(TSELSTest, case_uint32_uint8_2x8_2x32_2x8_2x8)
{
    test_tsels<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint32_uint16_2x8_2x16_2x8_2x8)
{
    test_tsels<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint32_uint32_2x8_2x8_2x8_2x8)
{
    test_tsels<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_half_uint8_2x16_2x32_2x16_2x16)
{
    test_tsels<aclFloat16, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_half_uint16_2x16_2x16_2x16_2x16)
{
    test_tsels<aclFloat16, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_half_uint32_2x16_2x8_2x16_2x16)
{
    test_tsels<aclFloat16, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16, true>();
}
TEST_F(TSELSTest, case_float_uint8_2x8_2x32_2x8_2x8)
{
    test_tsels<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_float_uint16_2x8_2x16_2x8_2x8)
{
    test_tsels<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_float_uint32_2x8_2x8_2x8_2x8)
{
    test_tsels<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>();
}
TEST_F(TSELSTest, case_uint8_uint8_2x32_2x64_2x128_2x31)
{
    test_tsels<uint8_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_uint16_uint8_2x32_2x64_2x128_2x31)
{
    test_tsels<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_float_uint8_2x32_2x64_2x128_2x31)
{
    test_tsels<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>();
}
TEST_F(TSELSTest, case_uint8_uint8_32x672_32x96_32x672_32x666)
{
    test_tsels<uint8_t, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>();
}
TEST_F(TSELSTest, case_half_uint8_32x672_32x96_32x672_32x666)
{
    test_tsels<aclFloat16, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666, true>();
}
TEST_F(TSELSTest, case_float_uint8_32x672_32x96_32x672_32x666)
{
    test_tsels<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>();
}
TEST_F(TSELSTest, case_float_uint8_1x8192_1x4096_1x8192_1x8192)
{
    test_tsels<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192>();
}
