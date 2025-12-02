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

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBFBQuant(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuant(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBVectorQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBVectorQuantDn(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuantDn(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class TMOVTest : public testing::Test {
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

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_nz2nd_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBNZ2ND<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_nz2nz_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBNZ2NZ<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_nz2dn_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBNZ2DN<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, typename QuantType, int32_t key>
void tmov_l0c2ub_fb_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);
    size_t FBQuantFileSize = N * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), FBQuantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, FBQuantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", FBQuantFileSize, src2Host, FBQuantFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, FBQuantFileSize, src2Host, FBQuantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBFBQuant<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_sc_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBSCQuant<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, typename QuantType, int32_t key>
void tmov_l0c2ub_nz2nz_vector_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);
    size_t FBQuantFileSize = N * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), FBQuantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, FBQuantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", FBQuantFileSize, src2Host, FBQuantFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, FBQuantFileSize, src2Host, FBQuantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBVectorQuantNz<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, typename QuantType, int32_t key>
void tmov_l0c2ub_nz2dn_vector_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);
    size_t FBQuantFileSize = N * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), FBQuantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, FBQuantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", FBQuantFileSize, src2Host, FBQuantFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, FBQuantFileSize, src2Host, FBQuantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBVectorQuantDn<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_nz2nz_sc_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBSCQuantNz<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename CType, typename AType, typename BType, int32_t key>
void tmov_l0c2ub_nz2dn_sc_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVL0c2UBSCQuantDn<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case_nz2nd_1)
{
    uint32_t M = 64;
    uint32_t K = 128;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_2)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nd_test<uint16_t, uint16_t, uint16_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_3)
{
    uint32_t M = 64;
    uint32_t K = 64;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_4)
{
    uint32_t M = 31;
    uint32_t K = 24;
    uint32_t N = 24;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_5)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 5>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_6)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nd_test<uint16_t, uint16_t, uint16_t, 6>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_7)
{
    uint32_t M = 64;
    uint32_t K = 32;
    uint32_t N = 32;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 7>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_8)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nd_test<uint32_t, uint16_t, uint16_t, 8>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_1)
{
    uint32_t M = 16;
    uint32_t K = 16;
    uint32_t N = 16;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint16_t, uint16_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_2)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint16_t, uint16_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_3)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint16_t, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_4)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint16_t, uint16_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_5)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint32_t, uint32_t, 5>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_6)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nz_test<uint16_t, uint16_t, uint16_t, 6>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_7)
{
    uint32_t M = 32;
    uint32_t K = 16;
    uint32_t N = 32;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint32_t, uint32_t, 7>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_8)
{
    uint32_t M = 128;
    uint32_t K = 16;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint16_t, uint16_t, 8>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_9)
{
    uint32_t M = 32;
    uint32_t K = 16;
    uint32_t N = 32;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint16_t, uint16_t, 9>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_10)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_test<uint32_t, uint32_t, uint32_t, 10>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_vector_quant_pre_1)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nz_vector_quant_test<int8_t, int8_t, int8_t, uint64_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_vector_quant_pre_2)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nz_vector_quant_test<uint16_t, int8_t, int8_t, uint64_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_vector_quant_pre_3)
{
    uint32_t M = 64;
    uint32_t K = 32;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nz_vector_quant_test<int8_t, uint32_t, uint32_t, uint64_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_vector_quant_pre_4)
{
    uint32_t M = 64;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_vector_quant_test<uint16_t, uint32_t, uint32_t, uint64_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_scalar_quant_pre_1)
{
    uint32_t M = 128;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_sc_quant_test<uint16_t, uint32_t, uint32_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_scalar_quant_pre_2)
{
    uint32_t M = 32;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_sc_quant_test<uint16_t, int8_t, int8_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_scalar_quant_pre_3)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 128;

    tmov_l0c2ub_nz2nz_sc_quant_test<int8_t, int8_t, int8_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_scalar_quant_pre_4)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_nz2nz_sc_quant_test<int8_t, uint32_t, uint32_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nd_vector_quant_1)
{
    tmov_l0c2ub_fb_quant_test<int8_t, int8_t, int8_t, uint64_t, 1>(32, 32, 128);
}

TEST_F(TMOVTest, case_nz2nd_vector_quant_2)
{
    tmov_l0c2ub_fb_quant_test<uint16_t, int8_t, uint8_t, uint64_t, 2>(32, 128, 32);
}

TEST_F(TMOVTest, case_nz2nd_vector_quant_3)
{
    tmov_l0c2ub_fb_quant_test<uint16_t, int8_t, uint8_t, uint64_t, 3>(128, 64, 96);
}

TEST_F(TMOVTest, case_nz2nd_vector_quant_4)
{
    tmov_l0c2ub_fb_quant_test<int8_t, uint32_t, uint32_t, uint64_t, 4>(112, 48, 96);
}

TEST_F(TMOVTest, case_nz2nd_vector_quant_5)
{
    tmov_l0c2ub_fb_quant_test<uint16_t, uint32_t, uint32_t, uint64_t, 5>(31, 128, 128);
}

TEST_F(TMOVTest, case_nz2nd_scalar_quant_1)
{
    tmov_l0c2ub_sc_quant_test<uint16_t, uint32_t, uint32_t, 1>(112, 48, 96);
}

TEST_F(TMOVTest, case_nz2nd_scalar_quant_2)
{
    tmov_l0c2ub_sc_quant_test<int8_t, uint32_t, uint32_t, 2>(112, 96, 64);
}

TEST_F(TMOVTest, case_nz2nd_scalar_quant_3)
{
    tmov_l0c2ub_sc_quant_test<uint16_t, int8_t, int8_t, 3>(32, 128, 64);
}

TEST_F(TMOVTest, case_nz2nd_scalar_quant_4)
{
    tmov_l0c2ub_sc_quant_test<int8_t, int8_t, int8_t, 4>(32, 32, 32);
}

TEST_F(TMOVTest, case_nz2dn_1)
{
    tmov_l0c2ub_nz2dn_test<uint32_t, uint32_t, uint32_t, 1>(64, 128, 32);
}

TEST_F(TMOVTest, case_nz2dn_2)
{
    tmov_l0c2ub_nz2dn_test<uint16_t, uint16_t, uint16_t, 2>(128, 32, 64);
}

TEST_F(TMOVTest, case_nz2dn_3)
{
    tmov_l0c2ub_nz2dn_test<uint16_t, uint16_t, uint16_t, 3>(48, 31, 31);
}

TEST_F(TMOVTest, case_nz2dn_4)
{
    tmov_l0c2ub_nz2dn_test<uint32_t, uint16_t, uint16_t, 4>(64, 128, 128);
}

TEST_F(TMOVTest, case_nz2dn_vector_quant_1)
{
    tmov_l0c2ub_nz2dn_vector_quant_test<int8_t, int8_t, int8_t, uint64_t, 1>(128, 128, 64);
}

TEST_F(TMOVTest, case_nz2dn_vector_quant_2)
{
    tmov_l0c2ub_nz2dn_vector_quant_test<uint16_t, int8_t, int8_t, uint64_t, 2>(32, 32, 128);
}

TEST_F(TMOVTest, case_nz2dn_vector_quant_3)
{
    tmov_l0c2ub_nz2dn_vector_quant_test<int8_t, uint16_t, uint16_t, uint64_t, 3>(128, 64, 128);
}

TEST_F(TMOVTest, case_nz2dn_vector_quant_4)
{
    tmov_l0c2ub_nz2dn_vector_quant_test<uint16_t, uint16_t, uint16_t, uint64_t, 4>(32, 32, 64);
}

TEST_F(TMOVTest, case_nz2dn_scalar_quant_1)
{
    tmov_l0c2ub_nz2dn_sc_quant_test<uint16_t, float, float, 1>(128, 32, 64);
}

TEST_F(TMOVTest, case_nz2dn_scalar_quant_2)
{
    tmov_l0c2ub_nz2dn_sc_quant_test<int8_t, float, float, 2>(128, 96, 64);
}

TEST_F(TMOVTest, case_nz2dn_scalar_quant_3)
{
    tmov_l0c2ub_nz2dn_sc_quant_test<uint16_t, int8_t, int8_t, 3>(32, 128, 64);
}

TEST_F(TMOVTest, case_nz2dn_scalar_quant_4)
{
    tmov_l0c2ub_nz2dn_sc_quant_test<int8_t, int8_t, int8_t, 4>(32, 32, 32);
}