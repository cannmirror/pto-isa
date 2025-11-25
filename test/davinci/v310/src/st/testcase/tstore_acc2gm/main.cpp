#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int format, int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void LaunchTStoreAcc2gm(uint8_t *out, uint8_t * src0, uint8_t *src1, void *stream);

template <typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void LaunchTStoreAcc2gmFp(uint8_t *out, uint8_t * src0, uint8_t *src1, uint8_t *quantTensor, void *stream);

class TStoreAcc2gmTest : public testing::Test {
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

template <int format, int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void test_tstore_acc2gm() {
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);

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
    LaunchTStoreAcc2gm<format,
        floatType,
        dstDataType,
        srcDataType,
        gShape0,
        gShape1,
        gShape2,
        gShape3,
        gShape4,
        gWholeShape0,
        gWholeShape1,
        gWholeShape2,
        gWholeShape3,
        gWholeShape4,
        validM,
        validN,
        validK,
        quantMode>(dstDevice, src0Device, src1Device, stream);

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

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int format, int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void test_tstore_acc2gmFp() {
    using ScalingT = uint64_t;
    constexpr int alignFbN = (validN * sizeof(ScalingT) + 127) / 128 * 128 / sizeof(ScalingT);

    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);
    size_t fpFileSize = alignFbN * sizeof(ScalingT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *quantTensorHost;
    uint8_t *dstDevice, *src0Device, *src1Device, *quantTensorDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&quantTensorHost), fpFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantTensorDevice, fpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_vector_gm.bin", fpFileSize, quantTensorHost, fpFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantTensorDevice, fpFileSize, quantTensorHost, fpFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmFp<
        dstDataType,
        srcDataType,
        gShape0,
        gShape1,
        gShape2,
        gShape3,
        gShape4,
        gWholeShape0,
        gWholeShape1,
        gWholeShape2,
        gWholeShape3,
        gWholeShape4,
        validM,
        validN,
        validK,
        quantMode>(dstDevice, src0Device, src1Device, quantTensorDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(quantTensorDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(quantTensorHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TStoreAcc2gmTest, case1)
{
    test_tstore_acc2gm<1, 0, float, float, 1, 1, 1, 128, 128, 1, 2, 3, 256, 128, 128, 128, 16, 0>();
}

TEST_F(TStoreAcc2gmTest, case2)
{
    test_tstore_acc2gm<1, 0, float, float, 1, 1, 1, 31, 32, 1, 2, 3, 31, 32, 31, 32, 15, 0>();
}

TEST_F(TStoreAcc2gmTest, case3)
{
    test_tstore_acc2gm<1, 0, float, uint16_t, 1, 1, 1, 65, 128, 1, 2, 3, 65, 128, 65, 128, 96, 0>();
}

TEST_F(TStoreAcc2gmTest, case4)
{
    test_tstore_acc2gm<1, 0, uint16_t, uint16_t, 1, 1, 1, 73, 64, 2, 2, 3, 73, 64, 73, 64, 32, 0>();
}

TEST_F(TStoreAcc2gmTest, case5)
{
    test_tstore_acc2gm<1, 1, float, uint16_t, 1, 1, 1, 13, 32, 2, 3, 7, 13, 32, 13, 32, 25, 0>();
}

TEST_F(TStoreAcc2gmTest, case6)
{
    test_tstore_acc2gm<1, 1, uint16_t, uint16_t, 1, 1, 1, 100, 222, 5, 7, 7, 100, 222, 100, 222, 60, 0>();
}

TEST_F(TStoreAcc2gmTest, case7)
{
    test_tstore_acc2gm<2, 0, float, float, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25, 0>();
}

TEST_F(TStoreAcc2gmTest, case8)
{
    test_tstore_acc2gm<2, 0, float, float, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45, 0>();
}

TEST_F(TStoreAcc2gmTest, case9)
{
    test_tstore_acc2gm<2, 0, float, uint16_t, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24, 0>();
}

TEST_F(TStoreAcc2gmTest, case10)
{
    test_tstore_acc2gm<2, 0, uint16_t, uint16_t, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23, 0>();
}

TEST_F(TStoreAcc2gmTest, case11)
{
    test_tstore_acc2gm<2, 1, float, uint16_t, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22, 0>();
}

TEST_F(TStoreAcc2gmTest, case12)
{
    test_tstore_acc2gm<2, 1, uint16_t, uint16_t, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32, 0>();
}

TEST_F(TStoreAcc2gmTest, case13)
{
    test_tstore_acc2gm<1, 0, int32_t, int8_t, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27, 0>();
}

TEST_F(TStoreAcc2gmTest, case14)
{
    test_tstore_acc2gm<2, 0, int32_t, int8_t, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30, 0>();
}

TEST_F(TStoreAcc2gmTest, case15)
{
    test_tstore_acc2gm<2, 0, float, float, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43, 0>();
}

TEST_F(TStoreAcc2gmTest, case16)
{
    test_tstore_acc2gm<1, 0, uint16_t, int8_t, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 1>();
}

TEST_F(TStoreAcc2gmTest, case17)
{
    test_tstore_acc2gmFp<2, 0, uint16_t, int8_t, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 2>();
}
