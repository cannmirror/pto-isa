#!/usr/bin/python3
# coding=utf-8

import os
import struct
import ctypes
import numpy as np
np.random.seed(19)


def gen_golden_data(case_name, param):
    src_type = param.datatype
    dst_type = param.datatype
    topk = param.topk // 2
    rows = param.row
    cols = param.src0_col // 2
    input_num = param.input_num
    # 获取每个输入的列数
    src_cols = [
        param.src0_col // 2, # 假设需要除以2
        param.src1_col // 2,
        param.src2_col // 2,
        param.src3_col // 2
    ]

    # 初始化空的列表来存储所有数据
    output_arr = []
    output_idx = []

    input_arr = np.random.uniform(low=0, high=1, size=(input_num * rows, cols)).astype(src_type) # num * rows*cols 随机浮点数组[-1,1]
    idx = np.arange(input_num * rows * cols).astype(np.uint32)
    # reshape to 32 cols (every sorted list)
    if input_num == 1:
        list_col = 32
    else:
        list_col = param.src0_col // 2

    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx.reshape(-1, list_col)
    # Sort each group of 32 elements based on input values in descending order
    sorted_indices = np.argsort(-input_reshaped, axis=1) # argsort() return idx
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)
    # print(f'排序后 sorted_input.shape: {sorted_input.shape}')
    # reshape back to rows, cols
    if input_num == 1 :
        output_arr = sorted_input
        output_idx = sorted_idx

        flat_input = sorted_input.flatten()
        flat_idx = sorted_idx.flatten()
        # Create pairs of (value, index)
        sorted_pairs = zip(flat_input, flat_idx)
        with open("input0.bin", 'wb') as f:
            for value, index in sorted_pairs:
                if src_type == np.float32:
                    # Packe the float32 value and the indx as a 32-bit unsigned integer
                    packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
                    f.write(packed_data)
                elif src_type == np.float16:
                    packed_data = struct.pack('e2xI', value, ctypes.c_uint32(index).value)
                    f.write(packed_data)
    else:
        for i in range(input_num):
            # 获取第i个输入的数据，根据每个list的col写入
            col_i = src_cols[i]
            # 展平数据
            flat_input_i = sorted_input[i,:cols].flatten()    
            flat_idx_i = sorted_idx[i,:cols].flatten()

            # 创建配对
            sorted_pairs_i = zip(flat_input_i, flat_idx_i)
            # 构造output的输出
            input_i = sorted_input[i,:col_i]
            idx_i = sorted_idx[i,:col_i]

            output_arr.append(input_i)
            output_idx.append(idx_i)

            # 生成文件名
            filename = f"input{i}.bin"

            # 写入bin文件
            with open(filename, 'wb') as f:
                for value, index in sorted_pairs_i:
                    if src_type == np.float32:
                        packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
                        f.write(packed_data)
                    elif src_type == np.float16:
                        packed_data = struct.pack('e2xI', value, ctypes.c_uint32(index).value)
                        f.write(packed_data)
    # flat_input_group = np.concatenate(output_arr).flatten()
    # flat_idx_group = np.concatenate(output_idx).flatten()

    # single case
    if case_name.startswith("TMRGSORTTest.case_single"):
        input_group = input_arr[0, :rows * cols // 128 * 128]
        idx_group = idx[:rows * cols // 128 * 128]
        single_output_reshape = input_group.reshape(-1, 128)
        single_idx_reshape = idx_group.reshape(-1, 128)
        single_sorted_indices = np.argsort(-single_output_reshape, axis=1)
        sorted_output_global = np.take_along_axis(single_output_reshape, single_sorted_indices, axis=1).flatten()
        sorted_idx_global = np.take_along_axis(single_idx_reshape, single_sorted_indices, axis=1).flatten()
        if rows % 128 != 0:
            zeros_output = np.zeros(cols * rows % 128, dtype=sorted_output_global.dtype)
            zeros_index = np.zeros(cols * rows % 128, dtype=np.uint32)
            single_sorted_output_global = np.concatenate((sorted_output_global, zeros_output))
            single_sorted_idx_global = np.concatenate((sorted_idx_global, zeros_index))
            sorted_pairs_global = zip(single_sorted_output_global, single_sorted_idx_global)
        else:
            sorted_pairs_global = zip(sorted_output_global, sorted_idx_global)
        write(sorted_pairs_global, src_type)
    else:
        flat_input_group = np.concatenate(output_arr).flatten()
        flat_idx_group = np.concatenate(output_idx).flatten()
        sorted_indices_global = np.argsort(-flat_input_group)
        sorted_output_global = flat_input_group[sorted_indices_global]
        sorted_idx_global = flat_idx_group[sorted_indices_global]
        prefixes = [
            "TMRGSORTTest.case_topk",
            "TMRGSORTTest.case_multi",
            "TMRGSORTTest.case_exhausted"
        ]
        if any(case_name.startswith(prefix) for prefix in prefixes):
            # 执行对应逻辑
            zeros_output = np.zeros(input_num * rows * cols - topk, dtype=sorted_output_global.dtype)
            zeros_index = np.zeros(input_num * rows * cols - topk, dtype=np.uint32)
            topk_sorted_output_global = np.concatenate((sorted_output_global[:topk], zeros_output))
            topk_sorted_idx_global = np.concatenate((sorted_idx_global[:topk], zeros_index))
            sorted_pairs_global = zip(topk_sorted_output_global, topk_sorted_idx_global)
            write(sorted_pairs_global, src_type)
        else:
            sorted_pairs_global = zip(sorted_output_global, sorted_idx_global)
            write(sorted_pairs_global, src_type)
    
def write(sorted_pairs_global, src_type):
    with open("golden.bin", 'wb') as f:
        for value, index in sorted_pairs_global:
            if src_type == np.float32:
                packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
                f.write(packed_data)
            elif src_type == np.float16:
                try:
                    packed_data = struct.pack('e2xI', value, ctypes.c_uint32(index).value)
                except:
                    value_bytes = struct.pack('e', value)
                    padding = b'\x00\x00'
                    index_bytes = struct.pack('I', ctypes.c_uint32(index).value)
                    packed_data = value_bytes + padding + index_bytes
                f.write(packed_data)


class tmrgsortParams:
    def __init__(self, datatype, row, src0_col, src1_col, src2_col, src3_col, input_num, topk):
        self.datatype = datatype
        self.row = row
        self.src0_col = src0_col
        self.src1_col = src1_col
        self.src2_col = src2_col
        self.src3_col = src3_col
        self.input_num = input_num
        self.topk = topk

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMRGSORTTest.case_multi1",
        "TMRGSORTTest.case_multi2",
        "TMRGSORTTest.case_multi3",
        "TMRGSORTTest.case_multi4",
        "TMRGSORTTest.case_exhausted1",
        "TMRGSORTTest.case_exhausted2",
        "TMRGSORTTest.case_single1",
        "TMRGSORTTest.case_single2",
        "TMRGSORTTest.case_single3",
        "TMRGSORTTest.case_single4",
        "TMRGSORTTest.case_single5",
        "TMRGSORTTest.case_single6",
        "TMRGSORTTest.case_single7",
        "TMRGSORTTest.case_single8",
        "TMRGSORTTest.case_topk1",
        "TMRGSORTTest.case_topk2",
        "TMRGSORTTest.case_topk3",
        "TMRGSORTTest.case_topk4",
        "TMRGSORTTest.case_topk5",
        "TMRGSORTTest.case_topk6",
        # 此名称需要和 TEST_F(TMATMULTest, case1)定义的名称一致
    ]

    case_params_list = [
        # TMRGSORTTest.case_multi, 多Tile
        tmrgsortParams(np.float32, 1, 128, 128, 128, 128, 4, 512),
        tmrgsortParams(np.float16, 1, 128, 128, 128, 128, 4, 512),
        tmrgsortParams(np.float32, 1, 128, 128, 128, 64, 4, 448),
        tmrgsortParams(np.float32, 1, 128, 128, 64, 0, 3, 128),
        # TMRGSORTTest.case_exhausted, 多Tile输入并开启耗尽模式
        tmrgsortParams(np.float32, 1, 64, 64, 0, 0, 2, 128),    # 多Tile输入，个数为2，且开启耗尽模式
        tmrgsortParams(np.float16, 1, 256, 256, 256, 0, 3, 768),
        # TMRGSORTTest.case_single, 单Tile(128个数字+128个索引)
        tmrgsortParams(np.float32, 1, 256, 0, 0, 0, 1, 0),  # reprat = 1, 且无尾块
        tmrgsortParams(np.float32, 1, 320, 0, 0, 0, 1, 0),  # reprat = 1, 尾块=64
        tmrgsortParams(np.float32, 1, 512, 0, 0, 0, 1, 0),  # reprat = 2, 且无尾块
        tmrgsortParams(np.float32, 1, 640, 0, 0, 0, 1, 0),  # reprat = 2, 尾块=128
        tmrgsortParams(np.float16, 1, 256, 0, 0, 0, 1, 0),  # reprat = 1, 且无尾块
        tmrgsortParams(np.float16, 1, 320, 0, 0, 0, 1, 0),  # reprat = 1, 尾块=64
        tmrgsortParams(np.float16, 1, 512, 0, 0, 0, 1, 0),  # reprat = 2, 且无尾块
        tmrgsortParams(np.float16, 1, 640, 0, 0, 0, 1, 0),  # reprat = 2, 尾块=128

        # TMRGSORTTest.case_topk，单Tile
        tmrgsortParams(np.float32, 1, 2048, 0, 0, 0, 1, 1024), # 16组64*4 single排序 ==> 8组256*4 single排序 ==> 1024 1024 multi排序,topk=1024
        tmrgsortParams(np.float32, 1, 2048, 0, 0, 0, 1, 2048), # 16组64*4 single排序 ==> 8组256*4 single排序 ==> 1024 1024 multi排序,topk=2048
        tmrgsortParams(np.float32, 1, 1280, 0, 0, 0, 1, 512), # 64*4 64*4 64*4 64*4 64*4 single排序 ==> 64*16 64*4 multi排序,topk=1280
        tmrgsortParams(np.float16, 1, 2048, 0, 0, 0, 1, 1024), # 16组64*4 single排序 ==> 8组256*4 single排序 ==> 1024 1024 multi排序,topk=1024
        tmrgsortParams(np.float16, 1, 2048, 0, 0, 0, 1, 2048), # 16组64*4 single排序 ==> 8组256*4 single排序 ==> 1024 1024 multi排序,topk=1024
        tmrgsortParams(np.float16, 1, 1280, 0, 0, 0, 1, 512), # 64*4 64*4 64*4 64*4 64*4 single排序 ==> 64*16 64*4 multi排序,topk=512
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)


