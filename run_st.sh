#!/bin/bash
set -e

if [ "$1" = "dailyBuild" ]; then
  # TCVT
  # A3
  echo "==================daily build, run all case================="
  python3 test/script/run_st.py -r sim -v a3 -t tcvt
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
  python3 test/script/run_st.py -r sim -v a3 -t tmatmul
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
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case7
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case8
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case9
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case10
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case11
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case12
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TMOVTest.case13
  # TEXTRACT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t textract
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case7
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case8
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case9
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case10
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case11
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case12
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case13
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case14
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case15
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case16

  # TMRGSORT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1

  # TSTORE
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tstore
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
  python3 test/script/run_st.py -r sim -v a3 -t trowsum
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test1
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test2
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test3

  # TROWEXPAND
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trowexpand
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case0
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case5

  # TGATHER
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tgather
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case2_int32
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case3_half
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case4_int16

  # TTRANS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t ttrans
  #A5
  python3 test/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case1

  # TSORT32
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsort32
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

  # TADD
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tadd
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_int32_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_int16_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_half_16x256_16x256_16x256

  # TCOPY
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tcopy

  # TSUB
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsub
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_int32_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_int16_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_half_16x256_16x256_16x256
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
  python3 test/script/run_st.py -r sim -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_param
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case1

  # TMRGSORT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1

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