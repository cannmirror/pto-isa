#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include "acl/acl.h"


using namespace std;
using namespace PtoTestCommon;

// //??? 为什么实例化这个模板函数？？

template <typename T, int KGRows_, int KGCols_, int KTRows_, int KTCols_, int reverse>
void LaunchTci(T *out, T S, void *stream);

// 1.定义一个测试夹具类
class TCITest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

// 2. 获取测试的输入输出以及golden所在的目录
std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// // 3. 又定义一次模板函数？？？   
// template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int S, int reverse>
// void LaunchTci(T *out, void *stream);

// 4. 定义测试逻辑函数：
template<typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse>
void test_tci(T S) {
    
    // 4.1 计算外层大小； 对于tci 感觉没必要
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    // 4.2 配置设备和数据流变量
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
     // 4.2 生命host侧变量和 device侧变量，并申请地址； 
    T *dstHost;
    T *dstDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 4.3 加载源数据到host侧输入-> 对于tci 接口不太需要；

    // ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    // ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    // 4.4 将host 侧数据搬运到 devices侧； -> 这个好像也不太需要；

    // aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4.5 调用算子接口进行计算：
    LaunchTci<T, kGRows_, kGCols_, kTRows_, kTCols_, reverse>(dstDevice, S, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    // aclrtFree(src0Device);
    // aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    // aclrtFreeHost(src0Host);
    // aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCITest, case1) {
    test_tci<int32_t, 1, 128, 1, 128, 1>(100);
}
TEST_F(TCITest, case2) {
    test_tci<int16_t, 1, 128, 1, 128, 0>(-1);
}
TEST_F(TCITest, case3) {
    test_tci<int16_t, 1, 128, 1, 128, 1>(-1);
}
TEST_F(TCITest, case4) {
    test_tci<int16_t, 1, 144, 1, 144, 1>(-1);
}
TEST_F(TCITest, case5) {
    test_tci<int32_t, 1, 132, 1, 144, 1>(-1);
}