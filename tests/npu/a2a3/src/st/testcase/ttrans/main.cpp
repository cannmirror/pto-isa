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

template <typename T, int tRows, int tCols, int vRows, int vCols>
void LaunchTTRANS(T *out, T *src, void *stream);

class TTRANSTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int tRows, int tCols, int vRows, int vCols>
void test_ttrans() {
    uint32_t M = tRows;
    uint32_t N = tCols;
    size_t srcFileSize = M * N * sizeof(T);
    size_t dstFileSize = M * N * sizeof(T);

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

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANS<T, tRows, tCols, vRows, vCols>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSTest, case1_float_16_8_16_8) {
    test_ttrans<float, 16, 8, 16, 8>();
}
TEST_F(TTRANSTest, case2_half_16_16_16_16) {
    test_ttrans<aclFloat16, 16, 16, 16, 16>();
}
TEST_F(TTRANSTest, case3_int8_32_32_32_32) {
    test_ttrans<uint8_t, 32, 32, 32, 32>();
}
TEST_F(TTRANSTest, case4_float_32_16_31_15) {
    test_ttrans<float, 32, 16, 31, 15>();
}
TEST_F(TTRANSTest, case5_half_32_32_31_31) {
    test_ttrans<aclFloat16, 32, 32, 31, 31>();
}
TEST_F(TTRANSTest, case6_int8_64_64_22_63) {
    test_ttrans<uint8_t, 64, 64, 22, 63>();
}
TEST_F(TTRANSTest, case7_float_2_512_2_512) {
    test_ttrans<float, 2, 512, 2, 512>();
}