#include "test_common.h"
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWSUMTestCase(void *out, void *src, aclrtStream stream);

class TROWSUMTest : public testing::Test {
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

template <uint32_t caseId, typename T, int row, int vaildRow, int srcCol, int srcVaildCol, int dstCol>
bool TRowSumTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = row * dstCol * sizeof(T);
    size_t srcByteSize = row * srcCol * sizeof(T);

    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;

    aclrtMallocHost(&dstHost, dstByteSize);
    aclrtMallocHost(&srcHost, srcByteSize);

    aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    launchTROWSUMTestCase<caseId>(dstDevice, srcDevice, stream);
    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstByteSize);
    std::vector<float> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp(golden, devFinal, 0.001f);
}

TEST_F(TROWSUMTest, case1)
{
    bool ret = TRowSumTestFramework<1, float, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case2)
{
    bool ret = TRowSumTestFramework<2, float, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case3)
{
    bool ret = TRowSumTestFramework<3, float, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case4)
{
    bool ret = TRowSumTestFramework<4, float, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case5)
{
    bool ret = TRowSumTestFramework<5, float, 7, 7, 448, 448, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case6)
{
    bool ret = TRowSumTestFramework<6, uint16_t, 256, 256, 16, 15, 1>();
    EXPECT_TRUE(ret);
}