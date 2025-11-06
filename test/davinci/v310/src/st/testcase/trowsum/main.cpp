#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

void launchTROWSUM_demo_float(float *out, float *src0, aclrtStream stream);

void launchTROWSUM_demo_half(uint16_t *out, uint16_t *src0, aclrtStream stream);

class TROWSUMTest : public testing::Test {
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

TEST_F(TROWSUMTest, test1)
{
    size_t fileSize = 16 * 16 * sizeof(float);
    size_t inputFileSize = fileSize;
    size_t outputFileSize = fileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    float *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&srcHost), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, srcHost, inputFileSize);

    aclrtMemcpy(srcDevice, inputFileSize, srcHost, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWSUM_demo_float(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, test2)
{
    size_t fileSize = 16 * 16 * sizeof(uint16_t);
    size_t inputFileSize = fileSize;
    size_t outputFileSize = fileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *dstHost, *srcHost;
    uint16_t *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&srcHost), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, srcHost, inputFileSize);

    aclrtMemcpy(srcDevice, inputFileSize, srcHost, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWSUM_demo_half(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, test3)
{
    size_t fileSize = 666 * 666 * sizeof(float);
    size_t inputFileSize = fileSize;
    size_t outputFileSize = fileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    float *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&srcHost), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, srcHost, inputFileSize);

    aclrtMemcpy(srcDevice, inputFileSize, srcHost, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWSUM_demo_float(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}