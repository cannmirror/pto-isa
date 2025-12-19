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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace PtoTestCommon;

template <int kRows, int kCols>
void LaunchTSEL(float *out, uint8_t *mask, float *src0, float *src1, void *stream);

class TSEL_Test : public testing::Test {
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

TEST_F(TSEL_Test, case_float_2x32)
{
    constexpr int kRows = 2;
    constexpr int kCols = 32;
    constexpr int kMaskCols = (kCols + 7) / 8;
    const size_t dataSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);
    const size_t maskSize = static_cast<size_t>(kRows) * static_cast<size_t>(kMaskCols) * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *src0Host, *src1Host;
    uint8_t *maskHost;
    float *dstDevice, *src0Device, *src1Device;
    uint8_t *maskDevice;
    aclrtMallocHost((void **)(&dstHost), dataSize);
    aclrtMallocHost((void **)(&src0Host), dataSize);
    aclrtMallocHost((void **)(&src1Host), dataSize);
    aclrtMallocHost((void **)(&maskHost), maskSize);
    aclrtMalloc((void **)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", readSize, maskHost, maskSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", readSize, src0Host, dataSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input3.bin", readSize, src1Host, dataSize));

    aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, dataSize, src0Host, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, dataSize, src1Host, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTSEL<kRows, kCols>(dstDevice, maskDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(maskDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(maskHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<float> out(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), dataSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), dataSize));
    EXPECT_TRUE(ResultCmp<float>(golden, out.data(), 0.0f));
}
