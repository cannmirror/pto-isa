/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*
 * Standalone driver for TFA (no gtest)
 */

#include <acl/acl.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "test_common.h"
#include "runtime/rt.h"

using namespace std;
using namespace PtoTestCommon;

static std::vector<std::string> Split(const std::string &s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

template<int S0, int HEAD_SIZE, int S1, int CUBE_S0 = S0, int CUBE_S1 = 128, bool INTERMEDIATE_CHECK = false>
void LaunchTFA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo, float *p_tile_fifo_fp32, float *global_sum_out, float *exp_max_out, float *o_out, float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo, aclrtStream stream);

// Track current case name for golden IO (replaces gtest naming)
static thread_local std::string g_case_name;

std::string GetGoldenDir() {
    return "./" + g_case_name;
}

/*
 * Template usage:
 * - The template parameter `INTERMEDIATE_CHECK` (default false) enables
 *   extra, more-detailed intermediate-value checks. When enabled, the
 *   host will compare the device softmax/intermediate tensor outputs
 *   (e.g. `p_out` / xexp) against golden files. On the device side the
 *   kernel should perform the necessary TSTORE operations to expose
 *   these intermediate buffers for host readback.
 *
 * Example:
 *   run_tfa<float, 64, 128, 256, true>(); // enable intermediate checks
 */
template<typename T, int S0, int HEAD_SIZE, int S1>
void run_tfa() {
    size_t fullSize = S0 * S1 * sizeof(T); // Keep output as float
    size_t qSize = S0 * HEAD_SIZE * sizeof(aclFloat16);
    size_t kSize = HEAD_SIZE * S1 * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *outHost;
    aclFloat16 *qHost, *kHost;
    aclFloat16 *xexpHost;
    float *tmpFloatExpHost;
    aclFloat16 *vHost;
    T *outDevice; // qk_out
    aclFloat16 *xexpDevice;
    T *midDevice = nullptr; // not used by this test but kept for symmetry
    aclFloat16 *qDevice, *kDevice;
    aclFloat16 *vDevice;
    T *out2Device; // pv_out
    T *out2Host;

    aclrtMallocHost((void **)(&outHost), fullSize); // Allocate output buffer
    aclrtMallocHost((void **)(&qHost), qSize);
    aclrtMallocHost((void **)(&kHost), kSize);

    aclrtMalloc((void **)&outDevice, fullSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
    size_t halfSize = S0 * S1 * sizeof(aclFloat16);
    size_t floatSize = S0 * S1 * sizeof(float);
    aclrtMalloc((void **)&xexpDevice, halfSize, ACL_MEM_MALLOC_HUGE_FIRST); // p_out (half)
    void *pOutFp32Device = nullptr;
    aclrtMalloc((void **)&pOutFp32Device, floatSize, ACL_MEM_MALLOC_HUGE_FIRST); // p_out_fp32 (float)
    // allocate v and out2 buffers
    size_t vSize = S1 * HEAD_SIZE * sizeof(aclFloat16);
    size_t pvPartSize = S0 * HEAD_SIZE * sizeof(T);
    int num_tiles = S1 / 128;
    size_t out2TotalSize = pvPartSize * num_tiles;
    aclrtMalloc((void **)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&out2Device, out2TotalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate global_sum buffer (per-tile S0 floats)
    size_t gsumTotalElems = static_cast<size_t>(S0) * static_cast<size_t>(num_tiles);
    size_t gsumSize = gsumTotalElems * sizeof(float);
    float *gSumDevice = nullptr;
    aclrtMalloc((void **)&gSumDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-tile exp_max buffer (per-tile S0 floats)
    float *expMaxDevice = nullptr;
    aclrtMalloc((void **)&expMaxDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate running output o (S0 x HEAD_SIZE)
    T *oDevice = nullptr;
    size_t oSize = pvPartSize; // S0 * HEAD_SIZE * sizeof(T)
    aclrtMalloc((void **)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-iteration running output snapshots (num_tiles * S0 * HEAD_SIZE)
    T *oPartsDevice = nullptr;
    size_t oPartsTotalSize = pvPartSize * num_tiles;
    aclrtMalloc((void **)&oPartsDevice, oPartsTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // write device buffer addresses/sizes for debug
    auto write_dev_entry = [](std::ofstream &ofs, const std::string &name, uint64_t addr, size_t bytes) {
        ofs << "[" << name << "]\n";
        ofs << "addr = \"0x" << std::hex << addr << "\"\n";
        ofs << std::dec;
        ofs << "size_bytes = " << bytes << "\n\n";
    };
    std::ofstream devToml("./device_addrs.toml", std::ios::out | std::ios::trunc);
    if (devToml.is_open()) {
        write_dev_entry(devToml, "q_device", reinterpret_cast<uint64_t>(qDevice), qSize);
        write_dev_entry(devToml, "k_device", reinterpret_cast<uint64_t>(kDevice), kSize);
        write_dev_entry(devToml, "v_device", reinterpret_cast<uint64_t>(vDevice), vSize);
        write_dev_entry(devToml, "qk_tile_fifo", reinterpret_cast<uint64_t>(outDevice), fullSize);
        write_dev_entry(devToml, "pv_tile_fifo", reinterpret_cast<uint64_t>(out2Device), out2TotalSize);
        write_dev_entry(devToml, "p_tile_fifo", reinterpret_cast<uint64_t>(xexpDevice), halfSize);
        write_dev_entry(devToml, "p_tile_fifo_fp32", reinterpret_cast<uint64_t>(pOutFp32Device), floatSize);
        write_dev_entry(devToml, "o_out", reinterpret_cast<uint64_t>(oDevice), oSize);
        devToml.close();
    }

    ReadFile(GetGoldenDir() + "/q.bin", qSize, qHost, qSize); // Read q data
    ReadFile(GetGoldenDir() + "/kt.bin", kSize, kHost, kSize);
    // read v
    aclrtMallocHost((void **)(&vHost), S1 * HEAD_SIZE * sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/v.bin", vSize, vHost, vSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Debug logging setup (preserve original tqksv behavior)
    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    // logging disabled

    std::cout << "[INFO] Intermediate checking is disabled" << std::endl;

    // Launch kernel, pass ffts ctrl addr and device-side log buffer, and xexp/tmp_float_exp device ptrs
    constexpr int CUBE_S0 = (S0 > 128 ? 128 : S0);
    constexpr int CUBE_S1 = (S0 <= 64 && (S1==1024 || S1==2048) ? 256 : 128);

    LaunchTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1>((uint16_t *)ffts, (aclFloat16*)qDevice, (aclFloat16*)kDevice, (aclFloat16*)vDevice, (aclFloat16*)xexpDevice, (float*)pOutFp32Device, (float*)gSumDevice, (float*)expMaxDevice, (float*)oDevice, (float*)oPartsDevice, (float*)outDevice, (float*)out2Device, stream);

    aclrtSynchronizeStream(stream);

    // copy outputs back
    aclrtMemcpy(outHost, fullSize, outDevice, fullSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMallocHost((void **)(&xexpHost), halfSize);
    aclrtMallocHost((void **)(&tmpFloatExpHost), floatSize);
    aclrtMemcpy(xexpHost, halfSize, xexpDevice, halfSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tmpFloatExpHost, floatSize, pOutFp32Device, floatSize, ACL_MEMCPY_DEVICE_TO_HOST);
    // copy second matmul partial outputs (concatenated per-tile)
    aclrtMallocHost((void **)(&out2Host), out2TotalSize);
    aclrtMemcpy(out2Host, out2TotalSize, out2Device, out2TotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy global_sum back
    float *gSumHost = nullptr;
    aclrtMallocHost((void **)(&gSumHost), gsumSize);
    aclrtMemcpy(gSumHost, gsumSize, gSumDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy exp_max back
    float *expMaxHost = nullptr;
    aclrtMallocHost((void **)(&expMaxHost), gsumSize);
    aclrtMemcpy(expMaxHost, gsumSize, expMaxDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy running output o back
    T *oHost = nullptr;
    aclrtMallocHost((void **)(&oHost), oSize);
    aclrtMemcpy(oHost, oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy per-iteration o parts back
    T *oPartsHost = nullptr;
    aclrtMallocHost((void **)(&oPartsHost), oPartsTotalSize);
    aclrtMemcpy(oPartsHost, oPartsTotalSize, oPartsDevice, oPartsTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/qk_out.bin", outHost, fullSize);
    WriteFile(GetGoldenDir() + "/p_out.bin", xexpHost, halfSize);
    WriteFile(GetGoldenDir() + "/p_out_fp32.bin", tmpFloatExpHost, floatSize);
    WriteFile(GetGoldenDir() + "/out2.bin", out2Host, out2TotalSize);
    // write per-tile global_sum parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(GetGoldenDir() + "/global_sum_part" + std::to_string(ti) + "_out.bin", gSumHost + partOffset, S0 * sizeof(float));
    }
    // write per-tile exp_max parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(GetGoldenDir() + "/exp_max_part" + std::to_string(ti) + "_out.bin", expMaxHost + partOffset, S0 * sizeof(float));
    }
    // write running output
    WriteFile(GetGoldenDir() + "/o_out.bin", oHost, oSize);
    // write per-iteration running output snapshots
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t byteOffset = static_cast<size_t>(ti) * pvPartSize;
        WriteFile(GetGoldenDir() + "/o_part" + std::to_string(ti) + "_out.bin", ((uint8_t*)oPartsHost) + byteOffset, pvPartSize);
    }

    aclrtFree(outDevice);
    aclrtFree(oDevice);
    aclrtFree(oPartsDevice);
    aclrtFree(qDevice);
    aclrtFree(kDevice);
    aclrtFree(xexpDevice);
    aclrtFree(pOutFp32Device);
    aclrtFree(vDevice);
    aclrtFree(out2Device);
    aclrtFree(gSumDevice);
    aclrtFree(expMaxDevice);

    // Final running output compare
    std::vector<float> golden_o(S0 * HEAD_SIZE);
    std::vector<float> dev_o(S0 * HEAD_SIZE);
    ReadFile(GetGoldenDir() + "/o.bin", oSize, golden_o.data(), oSize);
    ReadFile(GetGoldenDir() + "/o_out.bin", oSize, dev_o.data(), oSize);
    std::cout << "[CHECK] O running output compare" << std::endl;
    bool o_ok = ResultCmp<float>(golden_o, dev_o, 0.001f);

    std::cout << (o_ok ? "test success" : "test failed") << std::endl;


    aclrtFreeHost(outHost); // Free host memory
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(xexpHost);
    aclrtFreeHost(tmpFloatExpHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(out2Host);
    aclrtFreeHost(oHost);
    aclrtFreeHost(oPartsHost);
    aclrtFreeHost(gSumHost);
    aclrtFreeHost(expMaxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

}

template<typename T, int S0, int HEAD_SIZE, int S1>
void run_case(const std::string &case_name) {
    g_case_name = case_name;
    run_tfa<T, S0, HEAD_SIZE, S1>();
}

int main(int argc, char **argv) {
    struct CaseEntry {
        std::string name;
        std::function<void()> run;
    };

    std::vector<CaseEntry> cases = {
        {"case_float_H_128_S0_128_S1_1024", [](){ run_case<float, 128, 128, 1024>("case_float_H_128_S0_128_S1_1024"); }},
        {"case_float_H_128_S0_128_S1_2048", [](){ run_case<float, 128, 128, 2048>("case_float_H_128_S0_128_S1_2048"); }},
        {"case_float_H_128_S0_128_S1_8192", [](){ run_case<float, 128, 128, 8192>("case_float_H_128_S0_128_S1_8192"); }},
        {"case_float_H_128_S0_512_S1_1024", [](){ run_case<float, 512, 128, 1024>("case_float_H_128_S0_512_S1_1024"); }},
        {"case_float_H_128_S0_512_S1_2048", [](){ run_case<float, 512, 128, 2048>("case_float_H_128_S0_512_S1_2048"); }},
        {"case_float_H_128_S0_512_S1_8192", [](){ run_case<float, 512, 128, 8192>("case_float_H_128_S0_512_S1_8192"); }}
    };

    std::vector<std::string> filters;
    if (argc > 1) {
        std::string arg = argv[1];
        const std::string prefix = "--case=";
        if (arg.rfind(prefix, 0) == 0) {
            arg = arg.substr(prefix.size());
        }
        filters = Split(arg, ',');
    }

    if (!filters.empty()) {
        std::cout << "Running filtered cases: ";
        for (size_t i = 0; i < filters.size(); ++i) {
            std::cout << filters[i];
            if (i + 1 != filters.size()) std::cout << ",";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Running all cases" << std::endl;
    }

    auto should_run = [&](const std::string &name) {
        if (filters.empty()) return true;
        return std::find(filters.begin(), filters.end(), name) != filters.end();
    };

    for (const auto &c : cases) {
        if (should_run(c.name)) {
            c.run();
        }
    }

    return 0;
}

