#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>

using namespace std;
using namespace PtoTestCommon;

using DataType = float;

template <int32_t tilingKey>
void launchTMRGSORT_single_demo(float *out, float *src0, void* stream);
template <int32_t tilingKey>
void launchTMRGSORT_topk_demo(float *out, float *src0, void* stream);
template <int32_t tilingKey>
void launchTMRGSORT_multi_demo(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template <int32_t tilingKey>
void launchTMrgsort_demo_multi_exhausted(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);

class TMRGSORTTest : public testing::Test { 
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

template <int32_t tilingKey>
void tmrgsort_multi(uint16_t row, uint16_t col, uint16_t listNum) {
    size_t inputFileSize = row * col * sizeof(DataType); // uint16_t represent half
    size_t outputFileSize = listNum * row * col * sizeof(DataType); // uint16_t represent half
    std::cout << "Starting tmrgsort_multi with inputFileSize = " << inputFileSize << std::endl;
    std::cout << "Starting tmrgsort_multi with outputFileSize = " << outputFileSize << std::endl;
    // 使用vector动态管理设备/主机指针
    std::vector<DataType*> srcHostList;
    std::vector<DataType*> srcDeviceList;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost = nullptr, *tmpHost = nullptr;
    DataType *dstDevice = nullptr, *tmpDevice = nullptr;
    DataType *src0Host = nullptr, *src1Host = nullptr, *src2Host = nullptr, *src3Host = nullptr;
    DataType *src0Device = nullptr, *src1Device = nullptr, *src2Device = nullptr, *src3Device = nullptr;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMalloc((void**)(&dstDevice), outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    switch (listNum) {
    case 2:
        std::cout << "Processing case 2: listNum = 2" << std::endl;
        aclrtMallocHost((void **)(&src0Host), inputFileSize);
        aclrtMallocHost((void **)(&src1Host), inputFileSize);

        aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        //添加到管理列表
        srcHostList.push_back(src0Host);
        srcHostList.push_back(src1Host);
        srcDeviceList.push_back(src0Device);
        srcDeviceList.push_back(src1Device);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMRGSORT_multi_demo<tilingKey>(dstDevice, src0Device, src1Device, nullptr, nullptr, stream);
        break;
    case 3:
        std::cout << "Processing case 2: listNum = 3" << std::endl;
        aclrtMallocHost((void**)(&src0Host), inputFileSize);
        aclrtMallocHost((void**)(&src1Host), inputFileSize);
        aclrtMallocHost((void**)(&src2Host), inputFileSize);

        aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src2Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        srcHostList.push_back(src0Host);
        srcHostList.push_back(src1Host);
        srcHostList.push_back(src2Host);
        srcDeviceList.push_back(src0Device);
        srcDeviceList.push_back(src1Device);
        srcDeviceList.push_back(src2Device);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input2.bin", inputFileSize, src2Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src2Device, inputFileSize, src2Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMRGSORT_multi_demo<tilingKey>(dstDevice, src0Device, src1Device, src2Device, nullptr, stream);
        break;
    case 4:
        std::cout << "Processing case 2: listNum = 4" << std::endl;
        aclrtMallocHost((void**)(&src0Host), inputFileSize);
        aclrtMallocHost((void**)(&src1Host), inputFileSize);
        aclrtMallocHost((void**)(&src2Host), inputFileSize);
        aclrtMallocHost((void**)(&src3Host), inputFileSize);

        aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src2Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src3Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        srcHostList.push_back(src0Host);
        srcHostList.push_back(src1Host);
        srcHostList.push_back(src2Host);
        srcHostList.push_back(src3Host);
        srcDeviceList.push_back(src0Device);
        srcDeviceList.push_back(src1Device);
        srcDeviceList.push_back(src2Device);
        srcDeviceList.push_back(src3Device);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input2.bin", inputFileSize, src2Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input3.bin", inputFileSize, src3Host, inputFileSize);

        aclrtMemcpy(src3Device, inputFileSize, src3Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src2Device, inputFileSize, src2Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMRGSORT_multi_demo<tilingKey>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
        break;
    default:
        std::cerr << "Unsupported listNum: " << listNum << std::endl;
        break;
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    // 释放设备内存
    for (auto ptr : srcDeviceList) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    // 释放dstDevice(单独管理)
    if (dstDevice != nullptr) {
        aclrtFree(dstDevice);
    }
    
    // 释放主机内存
    for (auto ptr : srcHostList) {
        if (ptr != nullptr) {
            aclrtFreeHost(ptr);
        }
    }
    // 释放dstHost
    if (dstHost != nullptr) {
        aclrtFreeHost(dstHost);
    }
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

template <int32_t tilingKey>
void tmrgsort_exhausted(uint16_t row, uint16_t col, uint16_t listNum) {

    size_t inputFileSize = row * col * sizeof(DataType); // uint16_t represent half
    size_t outputFileSize = listNum * row * col * sizeof(DataType); // uint16_t represent half

    // 使用vector 动态管理设备/主机指针
    std::vector<DataType*> srcHostList;
    std::vector<DataType*> srcDeviceList;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost = nullptr, *tmpHost = nullptr;
    DataType *dstDevice = nullptr, *tmpDevice = nullptr;
    DataType *src0Host = nullptr, *src1Host = nullptr, *src2Host = nullptr, *src3Host = nullptr;
    DataType *src0Device = nullptr, *src1Device = nullptr, *src2Device = nullptr, *src3Device = nullptr;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMalloc((void**)(&dstDevice), outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    switch (listNum) {
    case 2:

        aclrtMallocHost((void **)(&src0Host), inputFileSize);
        aclrtMallocHost((void **)(&src1Host), inputFileSize);

        aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMrgsort_demo_multi_exhausted<tilingKey>(dstDevice, src0Device, src1Device, nullptr, nullptr, stream);
        break;
    case 3:
        aclrtMallocHost((void**)(&src0Host), inputFileSize);
        aclrtMallocHost((void**)(&src1Host), inputFileSize);
        aclrtMallocHost((void**)(&src2Host), inputFileSize);

        aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src2Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input2.bin", inputFileSize, src2Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src2Device, inputFileSize, src2Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMrgsort_demo_multi_exhausted<tilingKey>(dstDevice, src0Device, src1Device, src2Device, nullptr, stream);
        break;
    case 4:
        aclrtMallocHost((void**)(&src0Host), inputFileSize);
        aclrtMallocHost((void**)(&src1Host), inputFileSize);
        aclrtMallocHost((void**)(&src2Host), inputFileSize);
        aclrtMallocHost((void**)(&src3Host), inputFileSize);

        aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src2Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)&src3Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input2.bin", inputFileSize, src2Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input3.bin", inputFileSize, src3Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src2Device, inputFileSize, src2Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src3Device, inputFileSize, src3Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTMrgsort_demo_multi_exhausted<tilingKey>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
        break;
    default:
        std::cerr << "Unsupported listNum: " << listNum << std::endl;
        break;
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    // 释放设备内存
    for (auto ptr : srcDeviceList) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    // 释放dstDevice(单独管理)
    if (dstDevice != nullptr) {
        aclrtFree(dstDevice);
    }
    
    // 释放主机内存
    for (auto ptr : srcHostList) {
        if (ptr != nullptr) {
            aclrtFreeHost(ptr);
        }
    }
    // 释放dstHost
    if (dstHost != nullptr) {
        aclrtFreeHost(dstHost);
    }

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

template <int32_t tilingKey>
void tmrgsort_single(uint32_t row, uint32_t col) {
    size_t inputFileSize = row * col * sizeof(DataType);
    size_t outputFileSize = row * col * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *src0Host;
    DataType *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMRGSORT_single_demo<tilingKey>(dstDevice, src0Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t tilingKey>
void tmrgsort_topk(uint32_t row, uint32_t col) {
        size_t inputFileSize = row * col * sizeof(DataType);
    size_t outputFileSize = row * col * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *src0Host;
    DataType *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMRGSORT_topk_demo<tilingKey>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclrtFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMRGSORTTest, case_multi1)
{
    uint32_t row = 1;
    uint32_t col = 128;
    uint32_t listNum = 4;
    tmrgsort_multi<1>(row, col, listNum);

}

TEST_F(TMRGSORTTest, case_multi2)
{
    uint32_t row = 1;
    uint32_t col = 128;
    uint32_t listNum = 4;
    tmrgsort_multi<2>(row, col, listNum);

}

TEST_F(TMRGSORTTest, case_multi3)
{
    uint32_t row = 1;
    uint32_t col = 128;
    uint32_t listNum = 4;
    tmrgsort_multi<3>(row, col, listNum);

}

TEST_F(TMRGSORTTest, case_multi4)
{
    uint32_t row = 1;
    uint32_t col = 128;
    uint32_t listNum = 3;
    tmrgsort_multi<4>(row, col, listNum);

}

TEST_F(TMRGSORTTest, case_exhausted1)
{
    uint32_t row = 1;
    uint32_t col = 64;
    uint32_t listNum = 2;
    tmrgsort_exhausted<1>(row, col, listNum);
}

TEST_F(TMRGSORTTest, case_exhausted2)
{
    uint32_t row = 1;
    uint32_t col = 256;
    uint32_t listNum = 3;
    tmrgsort_exhausted<2>(row, col, listNum);
}

TEST_F(TMRGSORTTest, case_single1)
{
    tmrgsort_single<1>(1, 256);
}

TEST_F(TMRGSORTTest, case_single2)
{
    tmrgsort_single<2>(1, 320);
}

TEST_F(TMRGSORTTest, case_single3)
{
    tmrgsort_single<3>(1, 512);
}

TEST_F(TMRGSORTTest, case_single4)
{
    tmrgsort_single<4>(1, 640);
}

TEST_F(TMRGSORTTest, case_single5)
{
    tmrgsort_single<5>(1, 256);
}

TEST_F(TMRGSORTTest, case_single6)
{
    tmrgsort_single<6>(1, 320);
}

TEST_F(TMRGSORTTest, case_single7)
{
    tmrgsort_single<7>(1, 512);
}

TEST_F(TMRGSORTTest, case_single8)
{
    tmrgsort_single<8>(1, 640);
}

TEST_F(TMRGSORTTest, case_topk1) {
    tmrgsort_topk<1>(1, 2048);
}

TEST_F(TMRGSORTTest, case_topk2) {
    tmrgsort_topk<2>(1, 2048);
}

TEST_F(TMRGSORTTest, case_topk3) {
    tmrgsort_topk<3>(1, 1280);
}

TEST_F(TMRGSORTTest, case_topk4) {
    tmrgsort_topk<4>(1, 2048);
}

TEST_F(TMRGSORTTest, case_topk5) {
    tmrgsort_topk<5>(1, 2048);
}

TEST_F(TMRGSORTTest, case_topk6) {
    tmrgsort_topk<6>(1, 1280);
}