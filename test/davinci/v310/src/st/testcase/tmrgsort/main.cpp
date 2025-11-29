#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>

using namespace std;
using namespace PtoTestCommon;

using DataType = float;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void LanchTMrgsortMulti(DataType* out, DataType* src0, DataType* src1, DataType* src2, DataType* src3, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void LanchTMrgsortExhausted(DataType *out, DataType *src0, DataType *src1, DataType *src2, DataType *src3, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void LanchTMrgsortSingle(DataType* out, DataType* src, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void LanchTMrgsortTopK(DataType* out, DataType* src, void* stream);

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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void tmrgsort_multi() {
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = LISTNUM * kGRows_ * kGCols_ * sizeof(DataType);
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

    switch (LISTNUM) {
    case 2:
        std::cout << "Processing case 2: LISTNUM = 2" << std::endl;
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

        LanchTMrgsortMulti<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM>(
            dstDevice, src0Device, src1Device, nullptr, nullptr, stream);
        break;
    case 3:
        std::cout << "Processing case 2: LISTNUM = 3" << std::endl;
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

        LanchTMrgsortMulti<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM>(
            dstDevice, src0Device, src1Device, src2Device, nullptr, stream);
        break;
    case 4:
        std::cout << "Processing case 2: LISTNUM = 4" << std::endl;
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

        LanchTMrgsortMulti<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM>(
            dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
        break;
    default:
        std::cerr << "Unsupported LISTNUM: " << LISTNUM << std::endl;
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

    std::vector<T> golden(outputFileSize);
    std::vector<T> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void tmrgsort_exhausted() {

    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = LISTNUM * kGRows_ * kGCols_ * sizeof(DataType);

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

    switch (LISTNUM) {
    case 2:

        aclrtMallocHost((void **)(&src0Host), inputFileSize);
        aclrtMallocHost((void **)(&src1Host), inputFileSize);

        aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
        ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        LanchTMrgsortExhausted<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK,
                               LISTNUM>(dstDevice, src0Device, src1Device, nullptr, nullptr, stream);
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

        LanchTMrgsortExhausted<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK,
                               LISTNUM>(dstDevice, src0Device, src1Device, src2Device, nullptr, stream);
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

        LanchTMrgsortExhausted<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK,
                               LISTNUM>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
        break;
    default:
        std::cerr << "Unsupported LISTNUM: " << LISTNUM << std::endl;
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

    std::vector<DataType> golden(outputFileSize);
    std::vector<DataType> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void tmrgsort_single() {
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = kGRows_ * kGCols_ * sizeof(DataType);

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
    LanchTMrgsortSingle<T, kGRows_, kGCols_, kTRows_, kTCols_, blockLen>(dstDevice, src0Device, stream);

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

    std::vector<DataType> golden(outputFileSize);
    std::vector<DataType> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void tmrgsort_topk() {
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = kGRows_ * kGCols_ * sizeof(DataType);

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
    LanchTMrgsortTopK<T, kGRows_, kGCols_, kTRows_, kTCols_, topk>(dstDevice, src0Device, stream);

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

    std::vector<DataType> golden(outputFileSize);
    std::vector<DataType> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMRGSORTTest, case_multi1)
{
    tmrgsort_multi<float, 1, 128, 1, 128, 128, 128, 128, 512, 4>();
}

TEST_F(TMRGSORTTest, case_multi2)
{
    tmrgsort_multi<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4>();
}

TEST_F(TMRGSORTTest, case_multi3)
{
    tmrgsort_multi<float, 1, 128, 1, 128, 128, 128, 64, 448, 4>();
}

TEST_F(TMRGSORTTest, case_multi4)
{
    tmrgsort_multi<float, 1, 128, 1, 128, 128, 64, 0, 128, 3>();
}

TEST_F(TMRGSORTTest, case_exhausted1)
{
    tmrgsort_exhausted<float, 1, 64, 1, 64, 64, 0, 0, 128, 2>();
}

TEST_F(TMRGSORTTest, case_exhausted2)
{
    tmrgsort_exhausted<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3>();
}

TEST_F(TMRGSORTTest, case_single1)
{
    tmrgsort_single<float, 1, 256, 1, 256, 64>();
}

TEST_F(TMRGSORTTest, case_single2)
{
    tmrgsort_single<float, 1, 320, 1, 256, 64>();
}

TEST_F(TMRGSORTTest, case_single3)
{
    tmrgsort_single<float, 1, 512, 1, 512, 64>();
}

TEST_F(TMRGSORTTest, case_single4)
{
    tmrgsort_single<float, 1, 640, 1, 512, 64>();
}

TEST_F(TMRGSORTTest, case_single5)
{
    tmrgsort_single<uint16_t, 1, 256, 1, 256, 64>();
}

TEST_F(TMRGSORTTest, case_single6)
{
    tmrgsort_single<uint16_t, 1, 320, 1, 256, 64>();
}

TEST_F(TMRGSORTTest, case_single7)
{
    tmrgsort_single<uint16_t, 1, 512, 1, 512, 64>();
}

TEST_F(TMRGSORTTest, case_single8)
{
    tmrgsort_single<uint16_t, 1, 1024, 1, 1024, 256>();
}

TEST_F(TMRGSORTTest, case_topk1) {
    tmrgsort_topk<float, 1, 2048, 1, 2048, 1024>();
}

TEST_F(TMRGSORTTest, case_topk2) {
    tmrgsort_topk<float, 1, 2048, 1, 2048, 2048>();
}

TEST_F(TMRGSORTTest, case_topk3) {
    tmrgsort_topk<float, 1, 1280, 1, 1280, 512>();
}

TEST_F(TMRGSORTTest, case_topk4) {
    tmrgsort_topk<uint16_t, 1, 2048, 1, 2048, 1024>();
}

TEST_F(TMRGSORTTest, case_topk5) {
    tmrgsort_topk<uint16_t, 1, 2048, 1, 2048, 2048>();
}

TEST_F(TMRGSORTTest, case_topk6) {
    tmrgsort_topk<uint16_t, 1, 1280, 1, 1280, 512>();
}