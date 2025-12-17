/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef EVENT_HPP
#define EVENT_HPP

#include <pto/common/type.hpp>

namespace pto {
  enum class Op : uint16_t {
    SCALAR,
#ifdef __DAV_VEC__
    VECTOR,
    TADD,
    TADDS,
    TSUB,
    TMUL,
    TMULS,
    TDIV,
    TDIVS,
    TMIN,
    TMINS,
    TMAX,
    TSEL,
    TEXP,
    TSELS,
    TSQRT,
    TRSQRT,
    TEXPANDS,
    TPARTADD,
    TPARTMAX,
    TPARTMIN,
    TCMPS,
    TMRGSORT,
    TSORT32,
    TCI,
    TGATHER,
    TGATHERB,
    TCVT,
    TMOV_V2V,
    TROWSUM,
    TROWMAX,
    TROWMIN,
    TROWEXPAND,
    TCOLSUM,
    TCOLMAX,
    TCOLMIN,
    TTRANS,
    TSTORE_VEC,
    TLOAD,
#endif
#ifdef __DAV_CUBE__
    TMOV_M2B,
    TMOV_M2L,
    TMOV_M2R,
    TMOV_M2S,
    TMOV_A2V,
    TSTORE_ACC,
    TMATMUL,
    TMATMUL_ACC,
    TMATMUL_BIAS,
    TEXTRACT,
#endif
    OP_COUNT, // OpCode总数，新增OpCode请添加在OP_COUNT之前
  };

  template <Op OpCode>
  PTO_INTERNAL static constexpr pipe_t GetPipeByOp() {
    if constexpr (OpCode < Op::OP_COUNT) {
      return opPipeList[static_cast<uint16_t>(OpCode)];
    }
    return PIPE_ALL;
  }

  // 单流水线之间等待
  template <Op OpCode>
  PTO_INTERNAL void TSYNC_IMPL() {
    constexpr pipe_t pipe = GetPipeByOp<OpCode>();
    static_assert(pipe == PIPE_V, "Single Op TSYNC only supports Vector PTO Instruction.");
    pipe_barrier((pipe_t)pipe);
  }

  template <typename... WaitEvents>
  PTO_INTERNAL void waitAllEvents(WaitEvents&... events) {
    (events.Wait(), ...);
  }

  struct RecordEvent {};

#ifdef __CCE_AICORE__
  template <Op SrcOp, Op DstOp>
  struct Event {
    static constexpr pipe_t srcPipe = GetPipeByOp<SrcOp>();
    static constexpr pipe_t dstPipe = GetPipeByOp<DstOp>();
    static constexpr bool setIntraBlock =
      (srcPipe == PIPE_M) && (dstPipe == PIPE_V) ||
      (srcPipe == PIPE_V) && (dstPipe == PIPE_M);  // 标记该event跨核cube <-> Vec 根据不同平台修改set/wait指令

    CceEventIdType token = {nullptr};

    static_assert(srcPipe != PIPE_ALL, "SrcOp is invalid.");
    static_assert(dstPipe != PIPE_ALL, "DstOp is invalid.");
    static_assert(SrcOp != DstOp, "SrcOp is not allowed to be equal to DstOp.");
    static_assert(dstPipe != srcPipe, "SrcPipe is not allowed to be equal to dstPipe.");

    PTO_INTERNAL void Wait() {
      if (token.__opaque_token != nullptr) {
        if constexpr (setIntraBlock) {
          // CrossCoreEventWait
        } else {
          __pto_wait_flag(srcPipe, dstPipe, token);
        }
        token.__opaque_token = nullptr;
      }
    }

    PTO_INTERNAL void Record() {
      if constexpr (setIntraBlock) {
        // a2a3 wait->ffts a5 wait->set_intra_block
        // CrossCoreEventRecord
      } else {
        token = __pto_set_flag(srcPipe, dstPipe);
      }
    }

    // Event evt = OP(...)触发, 获取Event参数并自动record
    PTO_INTERNAL Event& operator=(RecordEvent) {
      Record();
      return *this;
    }

    // token 拷贝
    template <Op SOp, Op DOp>
    PTO_INTERNAL Event& operator=(const Event<SOp, DOp>& src) {
      token = src.token;
      return *this;
    }
  };
#endif
} // namespace pto
#endif