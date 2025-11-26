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
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case7
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case8
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case9
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case10
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case7
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case8
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case9
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case10
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULBIASTest.case11

  # TMOV
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmov
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
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias1
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias2
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias3
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias4
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias5
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias_dynamic6
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias_dynamic7
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias_dynamic8
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_fixpipe1
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_fixpipe2
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_fixpipe3
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_fixpipe4
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_fixpipe5
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_4
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_5
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_6
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_7
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_8
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_4
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_5
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_6
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_7
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_8
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_9
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_10
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_vector_quant_pre_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_vector_quant_pre_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_vector_quant_pre_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_vector_quant_pre_4
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_scalar_quant_pre_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_scalar_quant_pre_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_scalar_quant_pre_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nz_scalar_quant_pre_4
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_vector_quant_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_vector_quant_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_vector_quant_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_vector_quant_4
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_vector_quant_5
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_scalar_quant_1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_scalar_quant_2
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_scalar_quant_3
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_scalar_quant_4
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
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_multi1
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_multi2
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_multi3
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_multi4
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_exhausted1
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_exhausted2
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single1
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single2
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single3
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single4
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single6
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single7
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_single8
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk2
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk3
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk4
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk6

  # TSTORE
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tstore
  python3 test/script/run_st.py -r sim -v a3 -t tstore_acc2gm
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case6
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case7
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case8
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case9
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case10
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case11
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case12
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case1
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case3
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case5
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case9
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case11
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case13
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case16
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case17

  # TROWSUM
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trowsum
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test1
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test2
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test3

  # TCOLSUM
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t tcolsum
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test01
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test02
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test03
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test04
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test05
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test11
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test12
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test13
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test14
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test15
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test21
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test22
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test23
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test24
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test25

  # TCOLMAX
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t tcolmax
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test01
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test02
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test03
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test11
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test12
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test13
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test21
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test22
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test23
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test31
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test32
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test33
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test41
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test42
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test43
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test51
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test52
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test53
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test61
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test62
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test63
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test71
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test72
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test73

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

  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P0101
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P1010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P0001
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P0010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P0100
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P1000
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_P1111
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float_int_P1010

  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P0101
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P1010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P0001
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P0010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P0100
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P1000
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_half_P1111

  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_U16_P0101
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_U16_P1010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_I16_P0001
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_I16_P0010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_U32_P0100
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_I32_P1000
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_I32_P1111

  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P0101
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P1010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P0001
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P0010
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P0100
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P1000
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_b8_P1111

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

  # TPARTADD
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tpartadd
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_8x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x8_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_8x64
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x8
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_half_8x48_8x16_8x48
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_half_8x768_8x512_8x768
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_int16_8x48_8x48_8x16
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_int32_64x64_8x64_64x64

  # TLOAD
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tload_gm2mat
  python3 test/script/run_st.py -r sim -v a3 -t tload
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_128_128_VT_128_128_BLK1
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_64_VT_256_64_BLK8
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_s16_GT_128_127_VT_128_128_BLK1_PADMAX
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_u8_GT_128_127_VT_128_128_BLK1_PADMIN
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_DYN
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_STC
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_int64_GT_128_128_VT_128_128_BLK1
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8

  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_128_128_half_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_128_128_int8_t_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_128_128_float_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_64_128_half_DN2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_63_127_half_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_128_128_float_ND2ND
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_37_126_int8_t_ND2ND
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_2_3_64_128_1_3_4_128_128_384_128_half_ND2ND
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_2_3_33_99_1_2_3_33_99_int8_t_ND2ND
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_51_123_1_1_1_64_128_64_128_float_DN2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_63_127_1_1_1_63_127_64_128_half_DN2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_128_128_1_1_1_128_128_128_128_float_DN2DN
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_37_126_1_1_1_37_126_64_126_int8_t_DN2DN
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_2_3_64_128_1_3_4_96_128_64_768_half_DN2DN
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.2_2_4_16_8_2_2_4_16_8_80_48_float_NZ2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND
  python3 test/script/run_st.py -r sim -v a5 -t tload_mix -g TLOADMIXTest.1_2_1_64_128_1_3_4_128_128_128_128_uint64_t_ND2ND

  # TADD
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tadd
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_int32_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_int16_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_half_16x256_16x256_16x256

  # TSELS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsels -g TSELSTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a3 -t tsels -g TSELSTest.case_half_16x256_16x256_16x256
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsels -g TSELSTest.case_float_60x60_64x64_60x60

  # TCOPY
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tcopy

  # TFILLPAD
  #A3
  python3 test/script/run_st.py -r sim -v a3 -t tfillpad

  # TMINS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmins
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmins

  # TSUB
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsub
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_int32_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_int16_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a5 -t tsub -g TSUBTest.case_half_16x256_16x256_16x256

  # TCI
  # A3
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case2
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case3
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case4
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case5

else
  echo "==================simple build, run one case================="
  # TCI
  # A3
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tci -g TCITest.case5

  # TGATHERB
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tgatherb
  # A5

  # TCVT
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tcvt -g TCVTTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.case1

  # TMATMUL
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1

  # TMOV
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmov -g TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmov -g TMOVTest.case_bias1
  python3 test/script/run_st.py -r sim -v a5 -t tmov_l0c2ub -g TMOVTest.case_nz2nd_1
  # TEXTRACT
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_param
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t textract -g TEXTRACTTest.case1

  # TMRGSORT
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1

  # TMUL
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tmul -g TMULTest.case_float_64x64_64x64_64x64

  # TSTORE
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
  python3 test/script/run_st.py -r npu -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case1
  python3 test/script/build_st.py -r sim -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case17 
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tstore -g TStoreTest.case1

  # TCMPS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64

  # TROWSUM
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t trowsum -g TROWSUMTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowsum -g TROWSUMTest.test1

  # TCOLSUM
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t tcolsum -g TCOLMAXTest.test1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcolsum -g TCOLSUMTest.test01

  # TCOLMAX
  # A3
  # python3 test/script/run_st.py -r sim -v a3 -t tcolmax -g TCOLMAXTest.test1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tcolmax -g TCOLMAXTest.test01

  # TROWEXPAND
  # A3

  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trowexpand -g TROWEXPANDTest.case0

  # TGATHER
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tgather -g TGATHERTest.case1_float_P0101
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tgather -g TGATHERTest.case1_float

  # TTRANS
  # A3
  # python3 test/script/run_st.py -r npu -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8_param
  #A5
  python3 test/script/run_st.py -r sim -v a5 -t ttrans -g TTRANSTest.case1

  # TSELS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsels -g TSELSTest.case_float_64x64_64x64_64x64
  python3 test/script/run_st.py -r sim -v a3 -t tsels -g TSELSTest.case_half_16x256_16x256_16x256
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsels -g TSELSTest.case_float_16x200_20x224_16x200
  python3 test/script/run_st.py -r sim -v a5 -t tsels -g TSELSTest.case_half_2x32_2x32_2x32

  # TSORT32
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tsort32 -g TSort32Test.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case1

  # TADD
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64_64x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64

  # TPARTADD
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64

  # TFILLPAD
  #A3
  python3 test/script/run_st.py -r sim -v a3 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
  
  #A5
  # python3 test/script/run_st.py -r sim -v a5 -t tsort32 -g TSort32Test.case1

  # TMINS
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tmins
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmins -g TMINSTEST.case_float_60x60_64x64_60x60
  python3 test/script/run_st.py -r sim -v a5 -t tmins -g TMINSTEST.case_float_16x200_20x512_16x200
  python3 test/script/run_st.py -r sim -v a5 -t tmins -g TMINSTEST.case_float_1x3600_2x4096_1x3600

  # TLOAD
  # A3
  python3 test/script/run_st.py -r npu -v a3 -t tload_gm2mat -g TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX

  # TRSQRT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64

  # TSQRT
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64

  # TEXP
  # A3
  python3 test/script/run_st.py -r sim -v a3 -t texp -g TEXPTest.case_float_64x64_64x64_64x64
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t texp -g TEXPTest.case_float_64x64_64x64_64x64

  #TDIVS
  #A3
  python3 test/script/run_st.py -r sim -v a3 -t tdivs -g TDIVSTest.case1
  python3 test/script/run_st.py -r sim -v a3 -t tdivs -g TDIVSTest.case5
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tdivs -g TDIVSTest.case1
  python3 test/script/run_st.py -r sim -v a5 -t tdivs -g TDIVSTest.case5

  #TMULS
  #A3
  python3 test/script/run_st.py -r sim -v a3 -t tmuls -g TMULSTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tmuls -g TMULSTest.case1

  #TADDS
  #A3
  python3 test/script/run_st.py -r sim -v a3 -t tadds -g TADDSTest.case1
  # A5
  python3 test/script/run_st.py -r sim -v a5 -t tadds -g TADDSTest.case1
fi