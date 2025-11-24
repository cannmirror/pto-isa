#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

def save_to_binary_file(data, filename, dtype):
    """
    将数据保存成指定dtype的二进制文件;
    """
    np.array(data, dtype = dtype).tofile(filename)

def gen_golden_data_tci(case_name, param):
    dtype = param.dtype

    if param.reverse == 0:
        # 生成递增索引
        result = [param.begin + i for i in range(param.length)]
    elif param.reverse == 1:
        # 生成递减索引
        result = [param.begin - i for i in range(0,param.length)]
    print("####golden: ", result)
    save_to_binary_file(result, "golden.bin", dtype)
    save_to_binary_file(param.begin, "begin_index.bin", dtype)
    save_to_binary_file(param.reverse, "reverse.bin", dtype)


class tciParams:
    def __init__(self, dtype, begin, reverse, length, name):
        self.dtype = dtype
        self.begin = begin
        self.reverse = reverse
        self.length = length
        self.name = name

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TCITest.case_{dtype_str}_{param.begin}_{param.reverse}_{param.length}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("#####script_dir: ", script_dir)
    testcases_dir = os.path.join(script_dir, "testcases")
    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tciParams(np.int32, 100, 1, 128, "TCITest.case1"),
        tciParams(np.int16, -1, 0, 128, "TCITest.case2"),
        tciParams(np.int16, -1, 1, 128, "TCITest.case3"),
        tciParams(np.int16, -1, 1, 144, "TCITest.case4"),
        tciParams(np.int32, -1, 1, 132, "TCITest.case5"),
    ]

    for i, param in enumerate(case_params_list):
        case_name = param.name
        print(f"#####case_name: {case_name}")
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tci(case_name, param)
        os.chdir(original_dir)