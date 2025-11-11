#!/user/bin/python3
# coding=utf-8

import os

import numpy as np
np.random.seed(19)

def gen_golden_data(case_name, param):
    src_dtype = param.src_dtype
    dst_dtype = param.dst_dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]
    src = np.random.randint(1, 10, size=[H, W]).astype(src_dtype)
    golden = src.transpose((1, 0)).astype(dst_dtype)
    output = np.zeros([W, H]).astype(dst_dtype)
    for h in range(H):
        for w in range(W):
            if h >= h_valid or w >= w_valid:
                golden[w][h] = output[w][h]

    src.tofile("input.bin")
    golden.tofile("golden.bin")
    return output, src, golden

class TTRANSParams:
    def __init__(self, src_dtype, dst_dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.gloal_row = global_row
        self.gloal_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

if __name__ == "__main__":
    case_name_list = [
        "TTRANSTest.case1_float_16_8_16_8_param",
        "TTRANSTest.case2_half_16_16_16_16_param",
        "TTRANSTest.case3_int8_32_32_32_32_param",
        "TTRANSTest.case4_float_32_16_31_15_param",
        "TTRANSTest.case5_half_32_32_31_31_param",
        "TTRANSTest.case6_int8_64_64_22_63_param",
    ]

    case_params_list = [
        TTRANSParams(np.float32, np.float32, 16, 8, 16, 8, 16, 8),
        TTRANSParams(np.float16, np.float16, 16, 16, 16, 16, 16, 16),
        TTRANSParams(np.int8, np.int8, 32, 32, 32, 32, 32, 32),
        TTRANSParams(np.float32, np.float32, 32, 16, 32, 16, 31, 15),
        TTRANSParams(np.float16, np.float16, 32, 32, 32, 32, 31, 31),
        TTRANSParams(np.int8, np.int8, 64, 64, 64, 64, 22, 63),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)