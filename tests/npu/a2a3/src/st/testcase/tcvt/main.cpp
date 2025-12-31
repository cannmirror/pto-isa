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

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT(D *dst, S *src, void *stream);

class TCVTTest : public testing::Test {
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

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcvt()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t srcFileSize = M * N * sizeof(S);
    size_t dstFileSize = M * N * sizeof(D);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    D *dstHost, *dstDevice;
    S *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<D> golden(dstFileSize);
    std::vector<D> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<D>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// Macro to generate test cases for all shapes for a given type pair
#define GENERATE_TCVT_TESTS(dst_type, src_type, type_name) \
    TEST_F(TCVTTest, case_##type_name##_2x128) { test_tcvt<dst_type, src_type, 2, 128, 2, 128>(); } \
    TEST_F(TCVTTest, case_##type_name##_2x32) { test_tcvt<dst_type, src_type, 2, 32, 2, 32>(); } \
    TEST_F(TCVTTest, case_##type_name##_1x64) { test_tcvt<dst_type, src_type, 1, 64, 1, 64>(); } \
    TEST_F(TCVTTest, case_##type_name##_4x64) { test_tcvt<dst_type, src_type, 4, 64, 4, 64>(); }


// FP32 Source
GENERATE_TCVT_TESTS(float, float, fp32_fp32)
GENERATE_TCVT_TESTS(aclFloat16, float, fp32_fp16)
GENERATE_TCVT_TESTS(int32_t, float, fp32_int32)
GENERATE_TCVT_TESTS(int16_t, float, fp32_int16)
GENERATE_TCVT_TESTS(int64_t, float, fp32_int64)

// FP16 Source
GENERATE_TCVT_TESTS(float, aclFloat16, fp16_fp32)
GENERATE_TCVT_TESTS(int32_t, aclFloat16, fp16_int32)
GENERATE_TCVT_TESTS(int16_t, aclFloat16, fp16_int16)
GENERATE_TCVT_TESTS(int8_t, aclFloat16, fp16_int8)
GENERATE_TCVT_TESTS(uint8_t, aclFloat16, fp16_uint8)

// INT32 Source
GENERATE_TCVT_TESTS(float, int32_t, int32_fp32)
GENERATE_TCVT_TESTS(int16_t, int32_t, int32_int16)
GENERATE_TCVT_TESTS(int64_t, int32_t, int32_int64)

// INT16 Source
GENERATE_TCVT_TESTS(aclFloat16, int16_t, int16_fp16)
GENERATE_TCVT_TESTS(float, int16_t, int16_fp32)

// INT8 Source
GENERATE_TCVT_TESTS(aclFloat16, int8_t, int8_fp16)

// UINT8 Source
GENERATE_TCVT_TESTS(aclFloat16, uint8_t, uint8_fp16)

// INT64 Source
GENERATE_TCVT_TESTS(float, int64_t, int64_fp32)
GENERATE_TCVT_TESTS(int32_t, int64_t, int64_int32)
