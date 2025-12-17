/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_HPP
#define TSYNC_HPP

#include <pto/common/type.hpp>

namespace pto {
  // opPipeList maps each operation in Op enum to its corresponding pipeline type.
  // This array is used to determine which hardware pipeline should be used for each operation.
  constexpr pipe_t opPipeList[] = {
    PIPE_S,     // SCALAR
#ifdef __DAV_VEC__
    PIPE_V,     // VECTOR
    PIPE_V,     // TADD
    PIPE_V,     // TADDS
    PIPE_V,     // TSUB
    PIPE_V,     // TMUL
    PIPE_V,     // TMULS
    PIPE_V,     // TDIV
    PIPE_V,     // TDIVS
    PIPE_V,     // TMIN
    PIPE_V,     // TMINS
    PIPE_V,     // TMAX
    PIPE_V,     // TSEL
    PIPE_V,     // TEXP
    PIPE_V,     // TSELS
    PIPE_V,     // TSQRT
    PIPE_V,     // TRSQRT
    PIPE_V,     // TEXPANDS
    PIPE_V,     // TPARTADD
    PIPE_V,     // TPARTMAX
    PIPE_V,     // TPARTMIN
    PIPE_V,     // TCMPS
    PIPE_V,     // TMRGSORT
    PIPE_V,     // TSORT32
    PIPE_V,     // TCI
    PIPE_V,     // TGATHER
    PIPE_V,     // TGATHERB
    PIPE_V,     // TCVT
    PIPE_V,     // TMOV_V2V
    PIPE_V,     // TROWSUM
    PIPE_V,     // TROWMAX
    PIPE_V,     // TROWMIN
    PIPE_V,     // TROWEXPAND
    PIPE_V,     // TCOLSUM
    PIPE_V,     // TCOLMAX
    PIPE_V,     // TCOLMIN
    PIPE_V,     // TTRANS
    PIPE_MTE3,  // TSTORE_VEC
    PIPE_MTE2,  // TLOAD
#endif
#ifdef __DAV_CUBE__
    PIPE_MTE1,  // TMOV_M2B
    PIPE_MTE1,  // TMOV_M2L
    PIPE_MTE1,  // TMOV_M2R
    PIPE_FIX,   // TMOV_M2S
    PIPE_FIX,   // TMOV_A2V
    PIPE_FIX,   // TSTORE_ACC
    PIPE_M,     // TMATMUL
    PIPE_M,     // TMATMUL_ACC
    PIPE_M,     // TMATMUL_BIAS
    PIPE_MTE1,  // TEXTRACT
#endif
  };

  PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1) {
    return ((baseConst & 0xf) + ((mode & 0x3) << 4) + ((eventId & 0xf) << 8));
  }

  PTO_INTERNAL void CrossCoreEventRecord(pipe_t pipe, uint16_t eventId) {
#ifdef __DAV_CUBE__
    ffts_cross_core_sync(pipe, getFFTSMsg(0x2, eventId));
#endif
  }

  PTO_INTERNAL void CrossCoreEventWait(uint16_t eventId) {
#ifdef __DAV_VEC__
    wait_flag_dev(eventId);
#endif
  }
} // namespace pto
#endif