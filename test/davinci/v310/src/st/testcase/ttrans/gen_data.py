#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np
np.random.seed(19)

def gen_golden(case_name, param):
    srctype = param.srctype

    m, n = param.m, param.n

    x1_gm = np.random.randint(1, 5, [m, n]).astype(srctype)
    golden = x1_gm.transpose()
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")

class ttransParams:
    def __init__(self, srctype, m, n):
        self.srctype = srctype
        self.m = m
        self.n = n

if __name__ == "__main__":
    case_name_list = [
        "TTRANSTest.case1",
    ]

    case_params_list = [
        ttransParams(np.float32, 128 , 128),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)


