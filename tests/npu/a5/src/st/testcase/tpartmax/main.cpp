#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TPartMaxTest {

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void LaunchTPartMax(T *out, T *src0, T *src1, void *stream);

class TPARTMAXTest : public testing::Test {
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

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void test_tpartmax() {
    size_t src0FileSize = src0VR * src0VC * sizeof(T);
    size_t src1FileSize = src1VR * src1VC * sizeof(T);
    size_t dstFileSize  = dstVR  * dstVC  * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice,  dstFileSize,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPartMax<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPARTMAXTest, case_fp32_64x64_64x64_64x64) {
    test_tpartmax<float, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TPARTMAXTest, case_fp32_128x64_128x64_96x64) {
    test_tpartmax<float, 128, 64, 128, 64, 96, 64>();
}
TEST_F(TPARTMAXTest, case_fp32_95x95_95x95_95x95) {
    test_tpartmax<float, 95, 95, 95, 95, 95, 95>();
}
TEST_F(TPARTMAXTest, case_fp32_122x123_104x123_122x110) {
    test_tpartmax<float, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_fp16_122x123_104x123_122x110) {
    test_tpartmax<aclFloat16, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_s16_122x123_104x123_122x110) {
    test_tpartmax<int16_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_s32_122x123_104x123_122x110) {
    test_tpartmax<int32_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_u16_122x123_104x123_122x110) {
    test_tpartmax<uint16_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_u32_122x123_104x123_122x110) {
    test_tpartmax<uint32_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_u8_122x123_104x123_122x110) {
    test_tpartmax<uint8_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMAXTest, case_s8_122x123_104x123_122x110) {
    test_tpartmax<int8_t, 122, 123, 104, 123, 122, 110>();
}
} // namespace TPartMaxTest