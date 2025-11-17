#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTMOVL0c2UB(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

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

template <typename T, typename U, typename S, int32_t key>
void tmov_l0c2ub_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);

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
    launchTMOVL0c2UB<key>(dstDevice, src0Device, src1Device, stream);

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

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case_nd_1)
{
    uint32_t M = 64;
    uint32_t K = 128;
    uint32_t N = 128;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nd_2)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint16_t, uint16_t, uint16_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nd_3)
{
    uint32_t M = 64;
    uint32_t K = 64;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nd_4)
{
    uint32_t M = 31;
    uint32_t K = 24;
    uint32_t N = 24;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nd_5)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 5>(M, K, N);
}

TEST_F(TMOVTest, case_nd_6)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_l0c2ub_test<uint16_t, uint16_t, uint16_t, 6>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_1)
{
    uint32_t M = 16;
    uint32_t K = 16;
    uint32_t N = 16;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 7>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_2)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 8>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_3)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint16_t, uint16_t, uint16_t, 9>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_4)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint32_t, uint16_t, uint16_t, 10>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_5)
{
    uint32_t M = 128;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_l0c2ub_test<uint32_t, uint32_t, uint32_t, 11>(M, K, N);
}