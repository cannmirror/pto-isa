/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_IMPL_HPP
#define PTO_INSTR_IMPL_HPP

#include <pto/common/pto_tile.hpp>
#include <pto/common/type.hpp>

#ifdef MEMORY_BASE
#include "pto/npu/a2a3/TAssign.hpp"
#include "pto/npu/a2a3/TAdd.hpp"
#include "pto/npu/a2a3/TMins.hpp"
#include "pto/npu/a2a3/TAddS.hpp"
#include "pto/npu/a2a3/TDivS.hpp"
#include "pto/npu/a2a3/TMulS.hpp"
#include "pto/npu/a2a3/TSub.hpp"
#include "pto/npu/a2a3/TSels.hpp"
#include "pto/npu/a2a3/TMin.hpp"
#include "pto/npu/a2a3/TExpandS.hpp"
#include "pto/npu/a2a3/TMax.hpp"
#include "pto/npu/a2a3/TLoad.hpp"
#include "pto/npu/a2a3/TCvt.hpp"
#include "pto/npu/a2a3/TStore.hpp"
#include "pto/npu/a2a3/TTrans.hpp"
#include "pto/npu/a2a3/TRowSum.hpp"
#include "pto/npu/a2a3/TRowMax.hpp"
#include "pto/npu/a2a3/TRowMin.hpp"
#include "pto/npu/a2a3/TFillPad.hpp"
#include "pto/npu/a2a3/TColMax.hpp"
#include "pto/npu/a2a3/TMatmul.hpp"
#include "pto/npu/a2a3/TMrgSort.hpp"
#include "pto/npu/a2a3/TCmps.hpp"
#include "pto/npu/a2a3/TExtract.hpp"
#include "pto/npu/a2a3/TMov.hpp"
#include "pto/npu/a2a3/TMul.hpp"
#include "pto/npu/a2a3/TSort32.hpp"
#include "pto/npu/a2a3/TSel.hpp"
#include "pto/npu/a2a3/TGather.hpp"
#include "pto/npu/a2a3/TCvt.hpp"
#include "pto/npu/a2a3/TDiv.hpp"
#include "pto/npu/a2a3/TCopy.hpp"
#include "pto/npu/a2a3/TPartAdd.hpp"
#include "pto/npu/a2a3/TPartMax.hpp"
#include "pto/npu/a2a3/TPartMin.hpp"
#include "pto/npu/a2a3/TRowExpand.hpp"
#include "pto/npu/a2a3/TCI.hpp"
#include "pto/npu/a2a3/TColSum.hpp"
#include "pto/npu/a2a3/TUnaryOp.hpp"
#include "pto/npu/a2a3/TGatherB.hpp"
#include "pto/npu/a2a3/TColMin.hpp"
#endif

#ifdef REGISTER_BASE
#include "pto/npu/a5/TAssign.hpp"
#include "pto/npu/a5/TAdd.hpp"
#include "pto/npu/a5/TAddS.hpp"
#include "pto/npu/a5/TDivS.hpp"
#include "pto/npu/a5/TMulS.hpp"
#include "pto/npu/a5/TSub.hpp"
#include "pto/npu/a5/TMin.hpp"
#include "pto/npu/a5/TMax.hpp"
#include "pto/npu/a5/TLoad.hpp"
#include "pto/npu/a5/TCvt.hpp"
#include "pto/npu/a5/TStore.hpp"
#include "pto/npu/a5/TMrgSort.hpp"
#include "pto/npu/a5/TMatmul.hpp"
#include "pto/npu/a5/TCmps.hpp"
#include "pto/npu/a5/TColSum.hpp"
#include "pto/npu/a5/TColMax.hpp"
#include "pto/npu/a5/TRowSum.hpp"
#include "pto/npu/a5/TRowMax.hpp"
#include "pto/npu/a5/TFillPad.hpp"
#include "pto/npu/a5/TTrans.hpp"
#include "pto/npu/a5/Tci.hpp"
#include "pto/npu/a5/TSels.hpp"
#include "pto/npu/a5/TSel.hpp"
#include "pto/npu/a5/TSort32.hpp"
#include "pto/npu/a5/TExtract.hpp"
#include "pto/npu/a5/TMins.hpp"
#include "pto/npu/a5/TMov.hpp"
#include "pto/npu/a5/TRowExpand.hpp"
#include "pto/npu/a5/TCopy.hpp"
#include "pto/npu/a5/TPartAdd.hpp"
#include "pto/npu/a5/TPartMax.hpp"
#include "pto/npu/a5/TPartMin.hpp"
#include "pto/npu/a5/TGather.hpp"
#include "pto/npu/a5/TUnaryOp.hpp"
#include "pto/npu/a5/TGatherB.hpp"
#include "pto/npu/a5/TBinSOp.hpp"
#endif

#ifdef __CPU_SIM
    #include "pto/cpu/TSub.hpp"
    #include "pto/cpu/TMul.hpp"
    #include "pto/cpu/TDiv.hpp"
    #include "pto/cpu/TMatmul.hpp"
    #include "pto/cpu/TAssign.hpp"
    #include "pto/cpu/TAdd.hpp"
    #include "pto/cpu/TLoad.hpp"
    #include "pto/cpu/TStore.hpp"
    #include "pto/cpu/TExp.hpp"
    #include "pto/cpu/TRowmax.hpp"
    #include "pto/cpu/TMrgSort.hpp"
    #include "pto/cpu/TMov.hpp"
    #include "pto/cpu/TExtract.hpp"
    #include "pto/cpu/TRowSum.hpp"
    #include "pto/cpu/TMax.hpp"
    #include "pto/cpu/TExtract.hpp"
    #include "pto/cpu/TFillPad.hpp"
#endif

#endif