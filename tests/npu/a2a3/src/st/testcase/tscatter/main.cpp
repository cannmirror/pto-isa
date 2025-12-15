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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;


template <uint32_t caseId>
void launchTScatterTestCase(void *out, void *src, void *indexes, aclrtStream stream);

class TSCATTERTest : public testing::Test {
public: 
protected:
    void SetUp() override
    { 
    }

    void TearDown() override
    {
    }
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <uint32_t caseId, typename T, int row, int vaildRow, int col, int srcVaildCol>
bool TScatterTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = row * col * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    size_t indByteSize = row * col * sizeof(uint16_t);
    T *dstHost;
    T *srcHost;
    uint16_t *indHost;
    T *dstDevice;
    T *srcDevice;
    uint16_t *indDevice;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&indHost), indByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indDevice, indByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/indexes.bin", indByteSize, indHost, indByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(indDevice, indByteSize, indHost, indByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTScatterTestCase<caseId>(dstDevice, srcDevice, indDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(indDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(indHost);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp<T>(golden, devFinal, 0.001f);
}

TEST_F(TSCATTERTest, case1)
{
    bool ret = TScatterTestFramework<1, float, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case2)
{
    bool ret = TScatterTestFramework<2, aclFloat16, 63, 63, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case3)
{
    bool ret = TScatterTestFramework<3, int32_t, 31, 31, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case4)
{
    bool ret = TScatterTestFramework<4, int16_t, 15, 15, 192, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case5)
{
    bool ret = TScatterTestFramework<5, float, 7, 7, 448, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case6)
{
    bool ret = TScatterTestFramework<6, float, 256, 256, 16, 16>();
    EXPECT_TRUE(ret);
}

