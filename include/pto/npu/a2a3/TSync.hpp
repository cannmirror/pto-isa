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

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_VAL 0x2
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8
namespace pto {
  PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1) {
    return ((baseConst & FFTS_BASE_COUNT_WIDTH) +
      ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
      ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
  }

  PTO_INTERNAL void CrossCoreEventRecord(pipe_t pipe, uint16_t eventId) {
#ifdef __DAV_CUBE__
    ffts_cross_core_sync(pipe, getFFTSMsg(FFTS_MODE_VAL, eventId));
#endif
  }

  PTO_INTERNAL void CrossCoreEventWait(uint16_t eventId) {
#ifdef __DAV_VEC__
    wait_flag_dev(eventId);
#endif
  }
} // namespace pto
#endif