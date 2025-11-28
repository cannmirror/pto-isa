#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

def gen_golden_data(param):
    type = param.datatype
    input_arr0 = np.random.uniform(low=-8, high=8, size=(param.s0rows, param.s0cols)).astype(type)
    input_arr1 = np.random.uniform(low=-8, high=8, size=(param.s1rows, param.s1cols)).astype(type)
    output_arr = np.full((param.drows,param.dcols),-np.inf)
    output_arr[:param.s0rows,:param.s0cols] = input_arr0[:param.s0rows,:param.s0cols]
    output_arr[:param.s1rows,:param.s1cols] = input_arr1[:param.s1rows,:param.s1cols]
    src0 = input_arr0[0:min(param.s0rows,param.s1rows), 0:min(param.s0cols,param.s1cols)]
    src1 = input_arr0[0:min(param.s0rows,param.s1rows), 0:min(param.s0cols,param.s1cols)]
    res = src0 * (src0 > src1) + src1 * (src0 <= src1)
    rows, cols = res.shape
    output_arr[:rows, :cols] = res
    output_arr = output_arr.astype(type)
    input_arr0.tofile('input0.bin')
    input_arr1.tofile('input1.bin')
    output_arr.tofile('golden.bin')

class testParams:
    def __init__(self, name, datatype, dparam, s0param, s1param):
        self.name = name
        self.datatype = datatype
        self.drows, self.dcols, self.dRowStride = dparam
        self.s0rows, self.s0cols, self.s0RowStride = s0param
        self.s1rows, self.s1cols, self.s1RowStride = s1param

if __name__ == "__main__":
    case_list = [
        testParams('TPARTMAXTEST.test0', np.float32, (16, 32, 32), (16, 16, 16), (16, 32, 32)),
        testParams('TPARTMAXTEST.test1', np.float32, (22, 32, 32), (22, 32, 32), (16, 32, 32)),
        testParams('TPARTMAXTEST.test2', np.float32, (22, 40, 40), (22, 40, 40), (22, 32, 32)),
        testParams('TPARTMAXTEST.test3', np.float32, (22, 40, 40), (22, 40, 40), (8, 40, 40)),
        testParams('TPARTMAXTEST.test4', np.float32, (64, 128, 128), (64, 128, 128), (64, 128, 128)),
        testParams('TPARTMAXTEST.testEmpty0', np.float32, (16, 32, 32), (16, 0, 8), (16, 32, 32)),
        testParams('TPARTMAXTEST.testEmpty1', np.float32, (16, 32, 32), (0, 32, 32), (16, 32, 32)),
        testParams('TPARTMAXTEST.testEmpty0', np.float32, (16, 32, 32), (16, 32, 32), (16, 0, 8)),
        testParams('TPARTMAXTEST.testEmpty1', np.float32, (16, 32, 32), (16, 32, 32), (0, 32, 32)),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        orig_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(orig_dir)
