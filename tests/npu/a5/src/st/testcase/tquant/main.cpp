/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

#define DIV_ROUNDUP(a, b) (((a) + (b)-1) / (b))

using namespace std;
using namespace PtoTestCommon;

namespace TQuantTest {

template <int validRows, int validCols, int mode>
void LaunchTQuant(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);

class TQUANTTEST : public testing::Test {
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

template <int validRows, int validCols, int mode>
void test_tquant()
{
    size_t srcFileSize = validRows * validCols * sizeof(float);
    size_t dstExpFileSize = DIV_ROUNDUP(validRows * validCols, 32) * sizeof(uint8_t);
    size_t dstFileSize = validRows * validCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice;
    uint8_t *dstExpHost, *dstExpDevice;
    float *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&dstExpHost), dstExpFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstExpDevice, dstExpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQuant<validRows, validCols, mode>(dstExpDevice, dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstExpHost, dstExpFileSize, dstExpDevice, dstExpFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_e4m3.bin", dstHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_e8m0.bin", dstExpHost, dstExpFileSize);

    aclrtFree(dstDevice);
    aclrtFree(dstExpDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(dstExpHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden_fp8(dstFileSize);
    std::vector<uint8_t> dev_fp8(dstFileSize);
    std::vector<uint8_t> golden_e8m0(dstExpFileSize);
    std::vector<uint8_t> dev_e8m0(dstExpFileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8.bin", dstFileSize, golden_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_e8m0.bin", dstExpFileSize, golden_e8m0.data(), dstExpFileSize);
    ReadFile(GetGoldenDir() + "/output_e4m3.bin", dstFileSize, dev_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_e8m0.bin", dstExpFileSize, dev_e8m0.data(), dstExpFileSize);

    bool ret_fp8 = ResultCmp<uint8_t>(golden_fp8, dev_fp8, 0.0f);
    bool ret_e8m0 = ResultCmp<uint8_t>(golden_e8m0, dev_e8m0, 0.0f);

    EXPECT_TRUE(ret_e8m0);
    EXPECT_TRUE(ret_fp8);
}

TEST_F(TQUANTTEST, case_fp32_32x32_nd)
{
    test_tquant<32, 32, 0>();
}
TEST_F(TQUANTTEST, case_fp32_32x64_nd)
{
    test_tquant<32, 64, 0>();
}
TEST_F(TQUANTTEST, case_fp32_64x128_nd)
{
    test_tquant<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_fp32_128x128_nd)
{
    test_tquant<128, 128, 0>();
}
TEST_F(TQUANTTEST, case_fp32_32x64_nz)
{
    test_tquant<32, 64, 1>();
}
TEST_F(TQUANTTEST, case_fp32_64x128_nz)
{
    test_tquant<64, 128, 1>();
}
TEST_F(TQUANTTEST, case_fp32_128x128_nz)
{
    test_tquant<128, 128, 1>();
}
} // namespace TQuantTest