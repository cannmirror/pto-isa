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

#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

class TSQRTPLUSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool isInPlace = false>
void LaunchTSqrt(T *out, T *src, void *stream);

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool isInPlace = false>
void test_tsqrt()
{
    size_t srcFileSize = srcTileRow * srcTileCol * sizeof(T);
    size_t dstFileSize = dstTileRow * dstTileCol * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSqrt<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, isInPlace>(
        dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    float eps = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.0001f;
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TSQRTPLUSTest, case_float_64x64_64x128_64x64_inPlace_True) {
    test_tsqrt<float, 64, 64, 64, 128, 64, 64, true>();
}
TEST_F(TSQRTPLUSTest, case_float_128x64_64x64_64x64_inPlace_False) {
    test_tsqrt<float, 128, 64, 64, 64, 64, 64, false>();
}
TEST_F(TSQRTPLUSTest, case_half_64x64_128x128_64x64_inPlace_True) {
    test_tsqrt<aclFloat16, 64, 64, 128, 128, 64, 64, true>();
}
TEST_F(TSQRTPLUSTest, case_half_64x256_64x64_64x64_inPlace_False) {
    test_tsqrt<aclFloat16, 64, 256, 64, 64, 64, 64, false>();
}