#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX",
        "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADMAX_PADMAX",
        "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMIN_PADMAX_INPLACE",
        "TFILLPADTest.case_u16_GT_260_7_VT_260_32_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_s8_GT_260_7_VT_260_64_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMIN_PADMAX_EXPAND",
        "TFILLPADTest.case_s8_GT_259_7_VT_260_64_BLK1_PADMIN_PADMAX_EXPAND",
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        pass
        os.chdir(original_dir)

    pass
