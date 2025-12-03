#include "test_common.h"
#include <acl/acl.h>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWMAXTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir() {
  const testing::TestInfo *testInfo =
      testing::UnitTest::GetInstance()->current_test_info();
  const std::string caseName = testInfo->name();
  std::string suiteName = testInfo->test_suite_name();
  std::string fullPath = "../" + suiteName + "." + caseName;
  return fullPath;
}

class TROWMAXTest : public testing::Test {
public:
  aclrtStream stream;
  void *dstHost;
  void *srcHost;
  void *dstDevice;
  void *srcDevice;

protected:
  void SetUp() override {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&stream);
  }

  void TearDown() override {
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
  }

  template <typename T>
  bool CompareGolden(size_t dstByteSize, bool printAllEn = false) {
    std::vector<T> golden(dstByteSize);
    std::vector<T> result(dstByteSize);
    float eps = sizeof(T) == 4 ? 0.001f : 0.005f;
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(),
             dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(),
             dstByteSize);
    if (printAllEn) {
      return ResultCmp(golden, result, eps, 0, 1000, true);
    }
    return ResultCmp(golden, result, eps, 0, 1000, false, true);
  }

  template <uint32_t caseId, typename T, int row, int validRow, int srcCol,
            int srcVaildCol, int dstCol>
  bool TRowMaxTestFramework() {
    size_t dstByteSize = row * dstCol * sizeof(T);
    size_t srcByteSize = row * srcCol * sizeof(T);
    aclrtMallocHost(&dstHost, dstByteSize);
    aclrtMallocHost(&srcHost, srcByteSize);
    aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);

    launchTROWMAXTestCase<caseId>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize,
                ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost,
              validRow * dstCol * sizeof(T));

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    return CompareGolden<T>(dstByteSize);
  }
};

TEST_F(TROWMAXTest, case1) {
  bool ret = TRowMaxTestFramework<1, float, 127, 127, 64, 63, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case2) {
  bool ret = TRowMaxTestFramework<2, float, 63, 63, 64, 64, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case3) {
  bool ret = TRowMaxTestFramework<3, float, 31, 31, 128, 127, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case4) {
  bool ret = TRowMaxTestFramework<4, float, 15, 15, 192, 192, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case5) {
  bool ret = TRowMaxTestFramework<5, float, 7, 7, 448, 447, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case6) {
  bool ret = TRowMaxTestFramework<6, aclFloat16, 256, 256, 16, 15, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case7) {
  bool ret = TRowMaxTestFramework<7, float, 30, 30, 216, 216, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case8) {
  bool ret = TRowMaxTestFramework<8, float, 30, 30, 216, 24, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case9) {
  bool ret = TRowMaxTestFramework<9, float, 30, 11, 216, 216, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case10) {
  bool ret = TRowMaxTestFramework<10, float, 30, 11, 216, 24, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case11) {
  bool ret = TRowMaxTestFramework<11, float, 238, 238, 40, 40, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case12) {
  bool ret = TRowMaxTestFramework<12, float, 238, 238, 40, 16, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case13) {
  bool ret = TRowMaxTestFramework<13, float, 238, 121, 40, 40, 1>();
  EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case14) {
  bool ret = TRowMaxTestFramework<14, float, 238, 121, 40, 16, 1>();
  EXPECT_TRUE(ret);
}
