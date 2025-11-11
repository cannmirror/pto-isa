#!/usr/bin/python3
# coding=utf-8

import os
import struct
import ctypes
import numpy as np

np.random.seed(19)

def gen_golden_data(case_name, gInfo):
    data_type = gInfo.datatype
    gShape0 = gInfo.gShape0
    gShape1 = gInfo.gShape1
    gShape2 = gInfo.gShape2
    gShape3 = gInfo.gShape3
    gShape4 = gInfo.gShape4
    gWholeShape0 = gInfo.gWholeShape0
    gWholeShape1 = gInfo.gWholeShape1
    gWholeShape2 = gInfo.gWholeShape2
    gWholeShape3 = gInfo.gWholeShape3
    gWholeShape4 = gInfo.gWholeShape4

    if gInfo.format == "ND" or gInfo.format == "NZ":
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape3, gWholeShape4)).astype(data_type)
        output_arr = np.zeros(shape=(gWholeShape0, gWholeShape1,
                            gWholeShape2, gWholeShape3, gWholeShape4), dtype=data_type)
        output_arr[0:gShape0, 0: gShape1, 0: gShape2, 0: gShape3, 0: gShape4] \ 
                    = input_arr[0:gShape0, 0: gShape1, 0: gShape2, 0: gShape3, 0: gShape4]
    elif gInfo.format == "DN":
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                            gWholeShape2, gWholeShape4, gWholeShape3)).astype(data_type)
        output_arr = np.zeros(shape=(gWholeShape0, gWholeShape1,
                            gWholeShape2, gWholeShape4, gWholeShape3), dtype=data_type)
        output_arr[0:gShape0, 0: gShape1, 0: gShape2, 0: gShape4, 0: gShape3] \ 
                    = input_arr[0:gShape0, 0: gShape1, 0: gShape2, 0: gShape4, 0: gShape3]

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")

class GlobalTensorInfo:
    def __init__(self, datatype, format, gShape0, gShape1, gShape2, gShape3, gShape4,
                gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4):
        self.datatype = datatype
        self.format = format
        self.gShape0 = gShape0
        self.gShape1 = gShape1
        self.gShape2 = gShape2
        self.gShape3 = gShape3
        self.gShape4 = gShape4
        self.gWholeShape0 = gWholeShape0
        self.gWholeShape1 = gWholeShape1
        self.gWholeShape2 = gWholeShape2
        self.gWholeShape3 = gWholeShape3
        self.gWholeShape4 = gWholeShape4
        

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128",
        "TStoreTest.ND_int16_t_1_2_1_23_121_3_2_2_35_125",
        "TStoreTest.ND_int8_t_2_2_3_23_47_3_3_4_32_50",
        "TStoreTest.DN_float_1_1_1_4_21_1_1_1_8_32",
        "TStoreTest.DN_int16_t_3_1_1_1_124_5_1_1_2_128",
        "TStoreTest.DN_int8_t_2_1_2_32_32_3_4_3_64_35",
        "TStoreTest.NZ_float_1_1_1_16_8_1_1_2_16_8",
        "TStoreTest.NZ_int16_t_2_2_2_16_16_5_3_3_16_16",
        "TStoreTest.NZ_int8_t_1_2_1_16_32_2_4_2_16_32",
    ]

    case_params_list = [
        GlobalTensorInfo(np.float32, "ND", 1, 1, 1, 2, 128, 1, 1, 1, 2, 128),
        GlobalTensorInfo(np.int16, "ND", 1, 2, 1, 23, 121, 3, 2, 2, 35, 125),
        GlobalTensorInfo(np.int8, "ND", 2, 2, 3, 23, 47, 3, 3, 4, 32, 50),
        GlobalTensorInfo(np.float32, "DN", 1, 1, 1, 4, 21, 1, 1, 1, 8, 32),
        GlobalTensorInfo(np.int16, "DN", 3, 1, 1, 1, 124, 5, 1, 1, 2, 128),
        GlobalTensorInfo(np.int8, "DN", 2, 1, 2, 32, 32, 3, 4, 3, 64, 35),
        GlobalTensorInfo(np.float32, "NZ", 1, 1, 1, 16, 8, 1, 1, 2, 16, 8),
        GlobalTensorInfo(np.int16, "NZ", 2, 2, 2, 16, 16, 5, 3, 3, 16, 16),
        GlobalTensorInfo(np.int8, "NZ", 1, 2, 1, 16, 32, 2, 4, 2, 16, 32),
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)