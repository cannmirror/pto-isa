#!/bin/bash
set -e

if [ "$1" = "dailyBuild" ]; then
  # TCVT
  # A3
  echo "==================daily build, run all case================="
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case1
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case2
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case3
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case4
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case5
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case6
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case7
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case7

  # TMATMUL
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case1
  python3 test/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case2
  python3 test/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case3
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case6

  # TMOV
  # A3

  # A5

  # TEXTRACT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case1_half_0_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case2_int8_0_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case3_float_0_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case11_half_0_1_16_32_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case12_int8_0_1_48_64_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case13_float_0_1_32_48_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case21_half_1_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case22_int8_1_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case31_half_1_1_96_64_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case32_int8_1_1_32_32_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case41_dynamic_half_0_1_16_32_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case42_dynamic_int8_1_1_32_32_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case43_dynamic_int8_0_1_param
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case44_dynamic_half_1_1_param
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case1
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case2
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case3
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case4
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case6
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case7
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case8
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case9

  # TMRGSORT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_multi1
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_multi2
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_multi3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_multi4
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_exhausted1
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_exhausted2
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single1
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single2
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single4
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single5
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single6
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single7
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_single8
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk2
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk4
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk5
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk6
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk

  # TSTORE
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.ND_int16_t_1_2_1_23_121_3_2_2_35_125
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.ND_int8_t_2_2_3_23_47_3_3_4_32_50
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.DN_float_1_1_1_4_21_1_1_1_8_32
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.DN_int16_t_3_1_1_1_124_5_1_1_2_128
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.DN_int8_t_2_1_2_32_32_3_4_3_64_35
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.NZ_float_1_1_1_16_8_1_1_2_16_8
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.NZ_int16_t_2_2_2_16_16_5_3_3_16_16
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.NZ_int8_t_1_2_1_16_32_2_4_2_16_32
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_multi1
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_multi2
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_multi3
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_multi4
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_exhausted1
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_exhausted2
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single1
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single2
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single3
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single4
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single6
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single7
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_single8
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk1
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk2
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk3
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk4
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case_topk6


  # TROWSUM
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case1
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case2
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case3
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case4
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case5
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case6
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test1
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test2
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test3

  # TROWEXPAND
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand -g TROWEXPANDTest.case0
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand -g TROWEXPANDTest.case1
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand -g TROWEXPANDTest.case2
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand -g TROWEXPANDTest.case3
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand -g TROWEXPANDTest.case4
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case0
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case5

  # TGATHER
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P0101
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P1010
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P0001
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P0010
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P0100
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P1000
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P1111
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P0101
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P1010
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P0001
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P0010
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P0100
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P1000
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_half_P1111
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case_1D_float_32x1024_16x64
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case_1D_int32_32x512_16x256
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case_1D_half_16x1024_16x128
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case_1D_int16_32x256_32x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case2_int32
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case3_half
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case4_int16

  # TTRANS
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8_param
  #A5
  python3 test/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case1

  # TSORT32
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsort32 -g TSort32Test.case1
  python3 test/script/run_st.py -r sim -v a3 -t tsort32 -g TSort32Test.case2
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case1
  python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case2

  # TLOAD
  # A3&A5
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_128_128_VT_128_128_BLK1
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_64_VT_256_64_BLK8
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_s16_GT_128_127_VT_128_128_BLK1_PADMAX
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_u8_GT_128_127_VT_128_128_BLK1_PADMIN
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_DYN
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_STC
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
else
  echo "==================simple build, run one case================="
  # TCVT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tcvt -g TCVTTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case1

  # TMATMUL
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1

  # TMOV
  # A3

  # A5

  # TEXTRACT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTEST.case1_half_0_1_param
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTEST.case1

  # TMRGSORT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk

  # TSTORE
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case1

  # TROWSUM
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trowsum -g TROWSUMTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test1

  # TROWEXPAND
  # A3

  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case0

  # TGATHER
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tgather -g TGATHERTest.case1_float_P0101
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float

  # TTRANS
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8_param
  #A5
  python3 test/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case1

  # TSORT32
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsort32 -g TSort32Test.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case1

  #A5
  # python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case1

  # TLOAD
  # A3&A5
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
fi