#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    dst_type = param.ctype

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, False, False, False
    sFractalSize = param.sFractalSize if hasattr(param, 'sFractalSize') else 512
    dst_format = param.dst_format if hasattr(param, 'dst_format') else 'ND'

    x1_gm = np.random.randint(1, 5, [m, k]).astype(a_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(b_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(dst_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if is_bias:
        golden = np.matmul(x1_gm, x2_gm).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm, x2_gm).astype(dst_type)

    if dst_format == 'NZ':
        if dst_type == np.float32 and sFractalSize == 512:
            block_cols = 8
        else:
            block_cols = 16
    
        assert(m % 16) == 0, "M should be 16 aligned when matrix C is NZ format"
        assert(n % block_cols) == 0, "N should be aligned when matrix C is NZ format"
        golden = golden.reshape((int(m/16), 16, int(n/block_cols),block_cols)).transpose(2, 0, 1, 3).astype(dst_type)

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmovParams:
    def __init__(self, atype, btype, ctype, m, k, n, dst_format='ND', sFractalSize=512):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.dst_format = dst_format
        self.sFractalSize = sFractalSize

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMOVTest.case_nd_1",
        "TMOVTest.case_nd_2",
        "TMOVTest.case_nd_3",
        "TMOVTest.case_nd_4",
        "TMOVTest.case_nd_5",
        "TMOVTest.case_nd_6",
        "TMOVTest.case_nz2nz_1",
        "TMOVTest.case_nz2nz_2",
        "TMOVTest.case_nz2nz_3",
        "TMOVTest.case_nz2nz_4",
        "TMOVTest.case_nz2nz_5",
    ]

    case_params_list = [
        tmovParams(np.float16, np.float16, np.float32, 64, 128, 128),   # f32 -> f32
        tmovParams(np.float16, np.float16, np.float16, 128, 128, 64),    # f32 -> f16
        tmovParams(np.float16, np.float16, np.float32, 64, 64, 64),     # (64, 64) -> (64, 128)
        tmovParams(np.float16, np.float16, np.float32, 31, 24, 24),     # (31, 24)
        tmovParams(np.float16, np.float16, np.float32, 32, 32, 64),     # sub block id = 1
        tmovParams(np.float16, np.float16, bfloat16, 128, 64, 128),     # f32 -> bf16
        tmovParams(np.float16, np.float16, np.float32, 16, 16, 16, 'NZ'),  # nz2nz,f16@f16->f32
        tmovParams(np.float16, np.float16, np.float32, 128, 128, 64, 'NZ'),  # nz2nz,f16@f16->f32
        tmovParams(np.float16, np.float16, np.float16, 128, 128, 64, 'NZ'),  # nz2nz,f16@f16->f32->f16 in ub
        tmovParams(np.float16, np.float16, np.float32, 128, 128, 64, 'NZ', 1024), # nz2nz,f16@f16->f32  1024
        tmovParams(np.float32, np.float32, np.float32, 128, 128, 64, 'NZ'), # nz2nz,f32@f32->f32
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)