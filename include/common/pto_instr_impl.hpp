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
#include "davinci/v220/TAssign.hpp"
#include "davinci/v220/TAdd.hpp"
#include "davinci/v220/TMins.hpp"
#include "davinci/v220/TAddS.hpp"
#include "davinci/v220/TDivS.hpp"
#include "davinci/v220/TMulS.hpp"
#include "davinci/v220/TSub.hpp"
#include "davinci/v220/TSels.hpp"
#include "davinci/v220/TMin.hpp"
#include "davinci/v220/TLoad.hpp"
#include "davinci/v220/TCvt.hpp"
#include "davinci/v220/TStore.hpp"
#include "davinci/v220/TTrans.hpp"
#include "davinci/v220/TRowSum.hpp"
#include "davinci/v220/TRowMax.hpp"
#include "davinci/v220/TFillPad.hpp"
#include "davinci/v220/TColMax.hpp"
#include "davinci/v220/TMatmul.hpp"
#include "davinci/v220/TMrgSort.hpp"
#include "davinci/v220/TCmps.hpp"
#include "davinci/v220/TExtract.hpp"
#include "davinci/v220/TMov.hpp"
#include "davinci/v220/TMul.hpp"
#include "davinci/v220/TSort32.hpp"
#include "davinci/v220/TSel.hpp"
#include "davinci/v220/TGather.hpp"
#include "davinci/v220/TCvt.hpp"
#include "davinci/v220/TDiv.hpp"
#include "davinci/v220/TCopy.hpp"
#include "davinci/v220/TPartAdd.hpp"
#include "davinci/v220/TPartMax.hpp"
#include "davinci/v220/TPartMin.hpp"
#include "davinci/v220/TRowExpand.hpp"
#include "davinci/v220/TCI.hpp"
#include "davinci/v220/TColSum.hpp"
#include "davinci/v220/TUnaryOp.hpp"
#include "davinci/v220/TGatherB.hpp"
#endif

#ifdef __DAV_V310
#include "davinci/v310/TAssign.hpp"
#include "davinci/v310/TAdd.hpp"
#include "davinci/v310/TAddS.hpp"
#include "davinci/v310/TDivS.hpp"
#include "davinci/v310/TMulS.hpp"
#include "davinci/v310/TSub.hpp"
#include "davinci/v310/TMin.hpp"
#include "davinci/v310/TLoad.hpp"
#include "davinci/v310/TCvt.hpp"
#include "davinci/v310/TStore.hpp"
#include "davinci/v310/TMrgSort.hpp"
#include "davinci/v310/TMatmul.hpp"
#include "davinci/v310/TCmps.hpp"
#include "davinci/v310/TColSum.hpp"
#include "davinci/v310/TColMax.hpp"
#include "davinci/v310/TRowSum.hpp"
#include "davinci/v310/TRowMax.hpp"
#include "davinci/v310/TFillPad.hpp"
#include "davinci/v310/TTrans.hpp"
#include "davinci/v310/Tci.hpp"
#include "davinci/v310/TSels.hpp"
#include "davinci/v310/TSel.hpp"
#include "davinci/v310/TSort32.hpp"
#include "davinci/v310/TExtract.hpp"
#include "davinci/v310/TMins.hpp"
#include "davinci/v310/TMov.hpp"
#include "davinci/v310/TRowExpand.hpp"
#include "davinci/v310/TCopy.hpp"
#include "davinci/v310/TPartAdd.hpp"
#include "davinci/v310/TPartMax.hpp"
#include "davinci/v310/TPartMin.hpp"
#include "davinci/v310/TGather.hpp"
#include "davinci/v310/TUnaryOp.hpp"
#include "davinci/v310/TGatherB.hpp"
#include "davinci/v310/TBinSOp.hpp"
#endif

#ifdef __CPU_SIM
    #include "cpu_sim/TSub.hpp"
    #include "cpu_sim/TMul.hpp"
    #include "cpu_sim/TDiv.hpp"
    #include "cpu_sim/TMatmul.hpp"
    #include "cpu_sim/TAssign.hpp"
    #include "cpu_sim/TAdd.hpp"
    #include "cpu_sim/TLoad.hpp"
    #include "cpu_sim/TStore.hpp"
    #include "cpu_sim/TExp.hpp"
    #include "cpu_sim/TRowmax.hpp"
    #include "cpu_sim/TMrgSort.hpp"
    #include "cpu_sim/TMov.hpp"
    #include "cpu_sim/TExtract.hpp"
    #include "cpu_sim/TRowSum.hpp"
    #include "cpu_sim/TMax.hpp"
    #include "cpu_sim/TExtract.hpp"
    #include "cpu_sim/TFillPad.hpp"
#endif

#endif