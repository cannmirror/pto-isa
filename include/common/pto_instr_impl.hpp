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

#include "common/pto_tile.hpp"
#include "common/type.hpp"

#ifdef __DAV_V220
#include "npu/a2a3/TAssign.hpp"
#include "npu/a2a3/TAdd.hpp"
#include "npu/a2a3/TMins.hpp"
#include "npu/a2a3/TAddS.hpp"
#include "npu/a2a3/TDivS.hpp"
#include "npu/a2a3/TMulS.hpp"
#include "npu/a2a3/TSub.hpp"
#include "npu/a2a3/TSels.hpp"
#include "npu/a2a3/TMin.hpp"
#include "npu/a2a3/TMax.hpp"
#include "npu/a2a3/TLoad.hpp"
#include "npu/a2a3/TCvt.hpp"
#include "npu/a2a3/TStore.hpp"
#include "npu/a2a3/TTrans.hpp"
#include "npu/a2a3/TRowSum.hpp"
#include "npu/a2a3/TRowMax.hpp"
#include "npu/a2a3/TFillPad.hpp"
#include "npu/a2a3/TColMax.hpp"
#include "npu/a2a3/TMatmul.hpp"
#include "npu/a2a3/TMrgSort.hpp"
#include "npu/a2a3/TCmps.hpp"
#include "npu/a2a3/TExtract.hpp"
#include "npu/a2a3/TMov.hpp"
#include "npu/a2a3/TMul.hpp"
#include "npu/a2a3/TSort32.hpp"
#include "npu/a2a3/TSel.hpp"
#include "npu/a2a3/TGather.hpp"
#include "npu/a2a3/TCvt.hpp"
#include "npu/a2a3/TDiv.hpp"
#include "npu/a2a3/TCopy.hpp"
#include "npu/a2a3/TPartAdd.hpp"
#include "npu/a2a3/TPartMax.hpp"
#include "npu/a2a3/TPartMin.hpp"
#include "npu/a2a3/TRowExpand.hpp"
#include "npu/a2a3/TCI.hpp"
#include "npu/a2a3/TColSum.hpp"
#include "npu/a2a3/TUnaryOp.hpp"
#include "npu/a2a3/TGatherB.hpp"
#endif

#ifdef __DAV_V310
#include "npu/a5/TAssign.hpp"
#include "npu/a5/TAdd.hpp"
#include "npu/a5/TAddS.hpp"
#include "npu/a5/TDivS.hpp"
#include "npu/a5/TMulS.hpp"
#include "npu/a5/TSub.hpp"
#include "npu/a5/TMin.hpp"
#include "npu/a5/TMax.hpp"
#include "npu/a5/TLoad.hpp"
#include "npu/a5/TCvt.hpp"
#include "npu/a5/TStore.hpp"
#include "npu/a5/TMrgSort.hpp"
#include "npu/a5/TMatmul.hpp"
#include "npu/a5/TCmps.hpp"
#include "npu/a5/TColSum.hpp"
#include "npu/a5/TColMax.hpp"
#include "npu/a5/TRowSum.hpp"
#include "npu/a5/TRowMax.hpp"
#include "npu/a5/TFillPad.hpp"
#include "npu/a5/TTrans.hpp"
#include "npu/a5/Tci.hpp"
#include "npu/a5/TSels.hpp"
#include "npu/a5/TSel.hpp"
#include "npu/a5/TSort32.hpp"
#include "npu/a5/TExtract.hpp"
#include "npu/a5/TMins.hpp"
#include "npu/a5/TMov.hpp"
#include "npu/a5/TRowExpand.hpp"
#include "npu/a5/TCopy.hpp"
#include "npu/a5/TPartAdd.hpp"
#include "npu/a5/TPartMax.hpp"
#include "npu/a5/TPartMin.hpp"
#include "npu/a5/TGather.hpp"
#include "npu/a5/TUnaryOp.hpp"
#include "npu/a5/TGatherB.hpp"
#include "npu/a5/TBinSOp.hpp"
#endif

#ifdef __CPU_SIM
    #include "cpu/TSub.hpp"
    #include "cpu/TMul.hpp"
    #include "cpu/TDiv.hpp"
    #include "cpu/TMatmul.hpp"
    #include "cpu/TAssign.hpp"
    #include "cpu/TAdd.hpp"
    #include "cpu/TLoad.hpp"
    #include "cpu/TStore.hpp"
    #include "cpu/TExp.hpp"
    #include "cpu/TRowmax.hpp"
    #include "cpu/TMrgSort.hpp"
    #include "cpu/TMov.hpp"
    #include "cpu/TExtract.hpp"
    #include "cpu/TRowSum.hpp"
    #include "cpu/TMax.hpp"
    #include "cpu/TExtract.hpp"
    #include "cpu/TFillPad.hpp"
#endif

#endif