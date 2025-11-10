#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;


template <int32_t tilingKey>
void launchTROWEXPAND(uint8_t *out, uint8_t *src, void *stream);

class TROWEXPANDTest : public testing::Test {
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


TEST_F(TROWEXPANDTest, case0)
{
    uint32_t rows = 16;
    uint32_t srcCols = 16;
    uint32_t dstCols = 512;
    size_t inputFileSize = rows * srcCols * sizeof(uint16_t);
    size_t outputFileSize = rows * dstCols * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<0>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case1)
{
    uint32_t rows = 16;
    uint32_t srcCols = 32;
    uint32_t dstCols = 256;
    size_t inputFileSize = rows * srcCols * sizeof(uint8_t);
    size_t outputFileSize = rows * dstCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<1>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case2)
{
    uint32_t rows = 16;
    uint32_t srcCols = 8;
    uint32_t dstCols = 128;
    size_t inputFileSize = rows * srcCols * sizeof(float);
    size_t outputFileSize = rows * dstCols * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<2>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}


TEST_F(TROWEXPANDTest, case3)
{
    uint32_t rows = 16;
    uint32_t srcCols = 32 / sizeof(uint16_t);
    uint32_t dstCols = 512;
    size_t inputFileSize = rows * srcCols * sizeof(uint16_t);
    size_t outputFileSize = rows * dstCols * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<3>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case4)
{
    uint32_t rows = 16;
    uint32_t srcCols = 32;
    uint32_t dstCols = 256;
    size_t inputFileSize = rows * srcCols * sizeof(uint8_t);
    size_t outputFileSize = rows * dstCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<4>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case5)
{
    uint32_t rows = 16;
    uint32_t srcCols = 8;
    uint32_t dstCols = 128;
    size_t inputFileSize = rows * srcCols * sizeof(float);
    size_t outputFileSize = rows * dstCols * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    std::cout << "================================" << std::endl;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<5>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);


    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    std::cout << "================================" << std::endl;
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}







