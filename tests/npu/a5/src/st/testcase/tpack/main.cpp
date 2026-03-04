/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <type_traits>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TPackTest {

template <int validRows, int validCols, typename SrcType, typename DstType>
void LaunchTPack(DstType *dst, SrcType *src, void *stream);

class TPACKTEST : public testing::Test {
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

template <int validRows, int validCols, typename SrcType, typename DstType>
void test_tpack()
{
    size_t srcFileSize = validRows * validCols * sizeof(SrcType);
    size_t dstFileSize = validRows * validCols * sizeof(DstType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    SrcType *srcHost, *srcDevice;
    DstType *dstHost, *dstDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPack<validRows, validCols, SrcType, DstType>(dstDevice, srcDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree((void *)srcDevice);
    aclrtFree((void *)dstDevice);
    aclrtFreeHost((void *)srcHost);
    aclrtFreeHost((void *)dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare output against golden data as raw bytes (uint8_t) to avoid
    // floating-point NaN comparison issues — TPack is a bitwise operation.
    std::vector<uint8_t> golden(dstFileSize);
    std::vector<uint8_t> output(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, output.data(), dstFileSize);

    bool ret = ResultCmp<uint8_t>(golden, output, 0.0f);
    EXPECT_TRUE(ret);
}

// ---- b32 -> b16 cases ----
TEST_F(TPACKTEST, case_fp32_fp16_128x128)
{
    test_tpack<128, 128, uint32_t, uint16_t>();
}
TEST_F(TPACKTEST, case_fp32_bf16_128x128)
{
    test_tpack<128, 128, uint32_t, uint16_t>();
}
TEST_F(TPACKTEST, case_s32_s16_128x128)
{
    test_tpack<128, 128, uint32_t, uint16_t>();
}
TEST_F(TPACKTEST, case_u32_u16_128x128)
{
    test_tpack<128, 128, uint32_t, uint16_t>();
}

// ---- b32 -> b8 cases ----
TEST_F(TPACKTEST, case_fp32_fp8_128x128)
{
    test_tpack<128, 128, uint32_t, uint8_t>();
}
TEST_F(TPACKTEST, case_s32_s8_128x128)
{
    test_tpack<128, 128, uint32_t, uint8_t>();
}
TEST_F(TPACKTEST, case_u32_u8_128x128)
{
    test_tpack<128, 128, uint32_t, uint8_t>();
}

// ---- b16 -> b8 cases ----
TEST_F(TPACKTEST, case_fp16_fp8_128x128)
{
    test_tpack<128, 128, uint16_t, uint8_t>();
}
TEST_F(TPACKTEST, case_s16_s8_128x128)
{
    test_tpack<128, 128, uint16_t, uint8_t>();
}
TEST_F(TPACKTEST, case_u16_u8_128x128)
{
    test_tpack<128, 128, uint16_t, uint8_t>();
}

} // namespace TPackTest
