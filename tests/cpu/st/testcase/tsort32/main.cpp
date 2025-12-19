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

template <int kCols>
void LaunchTSORT32(float *outVal, uint32_t *outIdx, float *src, void *stream);

class TSORT32_Test : public testing::Test {
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

TEST_F(TSORT32_Test, case_float_1x32)
{
    constexpr int kCols = 32;
    const size_t srcSize = kCols * sizeof(float);
    const size_t valSize = kCols * sizeof(float);
    const size_t idxSize = kCols * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *srcHost, *valHost;
    uint32_t *idxHost;
    float *srcDevice, *valDevice;
    uint32_t *idxDevice;
    aclrtMallocHost((void **)(&srcHost), srcSize);
    aclrtMallocHost((void **)(&valHost), valSize);
    aclrtMallocHost((void **)(&idxHost), idxSize);
    aclrtMalloc((void **)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&valDevice, valSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, srcSize));
    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTSORT32<kCols>(valDevice, idxDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(valHost, valSize, valDevice, valSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(idxHost, idxSize, idxDevice, idxSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", valHost, valSize);
    WriteFile(GetGoldenDir() + "/output_idx.bin", idxHost, idxSize);

    aclrtFree(valDevice);
    aclrtFree(idxDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(valHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(kCols);
    std::vector<float> outVal(kCols);
    std::vector<uint32_t> goldenIdx(kCols);
    std::vector<uint32_t> outIdx(kCols);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), valSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden_idx.bin", readSize, goldenIdx.data(), idxSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, outVal.data(), valSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_idx.bin", readSize, outIdx.data(), idxSize));

    EXPECT_TRUE(ResultCmp<float>(golden, outVal.data(), 0.0f));
    EXPECT_TRUE(ResultCmp<uint32_t>(goldenIdx, outIdx.data(), 0.0f));
}
