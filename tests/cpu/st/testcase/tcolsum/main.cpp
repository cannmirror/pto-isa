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

template <typename T, int kRows, int kCols>
void LaunchTCOLSUM(T *out, T *src, bool isBinary, void *stream);

class TCOLSUM_Test : public testing::Test {
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

template <typename T, int kRows, int kCols>
static void run_case(bool isBinary)
{
    const size_t inSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(T);
    const size_t outSize = static_cast<size_t>(kCols) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), outSize);
    aclrtMallocHost((void **)(&srcHost), inSize);
    aclrtMalloc((void **)&dstDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, inSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, inSize));
    aclrtMemcpy(srcDevice, inSize, srcHost, inSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTCOLSUM<T, kRows, kCols>(dstDevice, srcDevice, isBinary, stream);
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

    std::vector<T> golden(static_cast<size_t>(kCols));
    std::vector<T> out(static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), outSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), outSize));
    EXPECT_TRUE(ResultCmp<T>(golden, out.data(), 0.001f));
}

TEST_F(TCOLSUM_Test, case_float_64x64)
{
    run_case<float, 64, 64>(false);
}
