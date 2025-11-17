#!/bin/bash
set -e

python3 test/script/build_st.py -r npu -v a3 -t tcvt -g TCVTTest.case1
python3 test/script/build_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 test/script/build_st.py -r npu -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_param
python3 test/script/build_st.py -r npu -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
# python3 test/script/run_st.py -r npu -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
python3 test/script/build_st.py -r npu -v a3 -t trowsum -g TROWSUMTest.case1
python3 test/script/build_st.py -r npu -v a3 -t tgather -g TGATHERTest.case1_float_P0101
python3 test/script/build_st.py -r npu -v a3 -t tsort32 -g TSort32Test.case1
python3 test/script/build_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64_64x64
python3 test/script/build_st.py -r npu -v a3 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64

python3 test/script/build_st.py -r npu -v a5 -t tcvt -g TCVTTest.case1
python3 test/script/build_st.py -r npu -v a5 -t tmatmul -g TMATMULTest.case1
python3 test/script/build_st.py -r npu -v a5 -t textract -g TEXTRACTTest.case1
python3 test/script/build_st.py -r npu -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
# python3 test/script/build_st.py -r npu -v a5 -t tstore -g TStoreTest.case1
python3 test/script/build_st.py -r npu -v a5 -t trowsum -g TROWSUMTest.test1
python3 test/script/build_st.py -r npu -v a5 -t trowexpand -g TROWEXPANDTest.case0
python3 test/script/build_st.py -r npu -v a5 -t tgather -g TGATHERTest.case1_float
python3 test/script/build_st.py -r npu -v a5 -t ttrans -g TTRANSTest.case1
python3 test/script/build_st.py -r npu -v a5 -t tsort32 -g TSort32Test.case1
python3 test/script/build_st.py -r npu -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
python3 test/script/build_st.py -r npu -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64
python3 test/script/build_st.py -r npu -v a5 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64
