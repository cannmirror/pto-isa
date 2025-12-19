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
void LaunchTCVT(int32_t *out, float *src, pto::RoundMode mode, void *stream);

class TCVT_Test : public testing::Test {
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

TEST_F(TCVT_Test, case_f32_to_i32_trunc_64x64)
{
    constexpr int kRows = 64;
    constexpr int kCols = 64;
    const size_t inSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);
    const size_t outSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *srcHost;
    int32_t *dstHost;
    float *srcDevice;
    int32_t *dstDevice;
    aclrtMallocHost((void **)(&srcHost), inSize);
    aclrtMallocHost((void **)(&dstHost), outSize);
    aclrtMalloc((void **)&srcDevice, inSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, inSize));
    aclrtMemcpy(srcDevice, inSize, srcHost, inSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTCVT<kRows, kCols>(dstDevice, srcDevice, pto::RoundMode::CAST_TRUNC, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outSize, dstDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<int32_t> out(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), outSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), outSize));
    EXPECT_TRUE(ResultCmp<int32_t>(golden, out.data(), 0.0f));
}
