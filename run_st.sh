#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -e

ENABLE_A3=false
ENABLE_A5=false
ENABLE_SIM=false
ENABLE_NPU=false
RUN_TYPE=sim

if [ "$1" = "a3" ]; then
  ENABLE_A3=true
elif [ "$1" = "a5" ]; then
  ENABLE_A5=true
elif [ "$1" = "a3_a5" ]; then
  ENABLE_A3=true
  ENABLE_A5=true
fi

if [ "$2" = "sim" ]; then
  RUN_TYPE=sim
elif [ "$2" = "npu" ]; then
  RUN_TYPE=npu
fi

if [ "$3" = "simple" ]; then
  ENABLE_SIMPLE=true
elif [ "$3" = "all" ]; then
  ENABLE_ALL=true
fi

if [ "$ENABLE_A3" = "true" ]; then                 # A2A3
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolsum -g TCOLSUMTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolmax -g TCOLMAXTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolmin -g TCOLMINTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tci -g TCITest.case1_int32
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcvt -g TCVTTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmov -g TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_param
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmul -g TMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tdiv -g TDIVTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t trowsum -g TROWSUMTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t trowexpand -g TROWEXPANDTest.case0
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tgather -g TGATHERTest.case1_float_P0101
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsels -g TSELSTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsels -g TSELSTest.case_half_16x256_16x256_16x256
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsort32 -g TSort32Test.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tadd -g TADDTest.case1_float_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsel -g TSELTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmins
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tload_gm2mat -g TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t texp -g TEXPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tdivs -g TDIVSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tdivs -g TDIVSTest.case5
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmuls -g TMULSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tadds -g TADDSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL


  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolsum
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolmax
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcolmin
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tcvt
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmatmul
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmov
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t textract
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmrgsort
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tstore
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tstore_acc2gm
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t trowsum
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t trowexpand
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tgather
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t ttrans
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsort32
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tpartadd
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsel
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tload_gm2mat
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tload
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tadd
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsels
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tmins
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tsub
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tci
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t tgatherb
    python3 test/script/run_st.py -r $RUN_TYPE -v a3 -t texp
    python3 test/script/run_st.py -r $RUN_TYPE -v a3 -t trsqrt
    python3 test/script/run_st.py -r $RUN_TYPE -v a3 -t tsqrt
    python3 tests/script/run_st.py -r $RUN_TYPE -v a3 -t texpands
  fi
fi

if [ "$ENABLE_A5" = "true" ]; then
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tci -g TCITest.case5
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcvt -g TCVTTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmatmul -g TMATMULTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmov -g TMOVTest.case_bias1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t textract -g TEXTRACTTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tstore -g TStoreTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trowsum -g TROWSUMTest.test1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcolsum -g TCOLSUMTest.test01
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcolmax -g TCOLMAXTest.test01
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trowexpand -g TROWEXPANDTest.case5_float_16_8_16_127
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tgather -g TGATHERTest.case1_float
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t ttrans -g TTRANSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsels -g TSELSTest.case_float_16x200_20x224_16x200
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsels -g TSELSTest.case_half_2x32_2x32_2x32
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsort32 -g TSort32Test.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tadd -g TADDTest.case1_float_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsort32 -g TSort32Test.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmins -g TMINSTEST.case_float_60x60_64x64_60x60
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmins -g TMINSTEST.case_float_16x200_20x512_16x200
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmins -g TMINSTEST.case_float_1x3600_2x4096_1x3600
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t texp -g TEXPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tdivs -g TDIVSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tdivs -g TDIVSTest.case5
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmuls -g TMULSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tadds -g TADDSTest.case1
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL


  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tgatherb
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tci
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcvt
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmatmul
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmov
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmov_l0c2ub
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t textract
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcmps
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmrgsort
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tstore
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trowsum
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcolsum
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tcolmax
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trowexpand
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tgather
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t ttrans
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsels
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsort32
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tadd
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tpartadd
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsort32
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmins
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tload
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t trsqrt
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tsqrt
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t texp
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tdivs
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tmuls
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t tadds
    python3 tests/script/run_st.py -r $RUN_TYPE -v a5 -t texpands
  fi
fi