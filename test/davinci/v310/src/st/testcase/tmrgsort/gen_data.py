#!/usr/bin/python3
# coding=utf-8

import os
import struct
import ctypes
import numpy as np
np.random.seed(19)

def find_and_zero(arr, tar):
    for idx, item in enumerate(arr):
        if not isinstance(item, (np.floating)):
            return -1
    if not all(isinstance(x, (np.floating)) for x in arr):
        raise ValueError("输入必须是一个数字列表")
    if not isinstance(tar, (np.floating)):
        return -1
    
    n = len(arr)
    for i in range(n-1, -1, -1):  # 从后往前遍历
        if arr[i] == tar:  # 根据你的条件修改这一行
            for j in range(i + 1, n):
                arr[j] = 0
            return i
    return -1

def zero_after_index(arr, i):
    # 检查索引是否合法
    if i < 0 or i >= len(arr):
        return
    
    # 将位置i之后的元素置为0
    for j in range(i + 1, len(arr)):
        arr[j] = 0

def gen_golden_data(param):
    src_type = param.datatype
    topk = param.topk // 2
    cols = param.src0_col // 2
    input_num = param.input_num
    case_name = param.case_name
    blockLen = param.blockLen // 2
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

    input_arr = np.random.uniform(low=0, high=1, size=(input_num, cols)).astype(src_type) # num * cols 随机浮点数组[-1,1]
    idx = np.arange(input_num * cols).astype(np.uint32)
    # reshape to 32 cols (every sorted list)
    if input_num == 1:
        list_col = blockLen
    else:
        list_col = cols

    last_data = [0] * input_num
    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx.reshape(-1, list_col)
    # Sort each group of 32 elements based on input values in descending order
    sorted_indices = np.argsort(-input_reshaped, kind = 'stable', axis=1) # argsort() return idx
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)
    # print(f'排序后 sorted_input.shape: {sorted_input.shape}')
    # reshape back to 1, cols
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
            last_data[i] = flat_input_i[len(flat_input_i) - 1]

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

    # single case
    if case_name.startswith("TMRGSORTTest.case_single"):
        blocksLen = list_col * 4
        input_group = input_arr[0, :cols // blocksLen * blocksLen]
        idx_group = idx[:cols // blocksLen * blocksLen]
        single_output_reshape = input_group.reshape(-1, blocksLen)
        single_idx_reshape = idx_group.reshape(-1, blocksLen)
        single_sorted_indices = np.argsort(-single_output_reshape, kind = 'stable', axis=1)
        sorted_output_global = np.take_along_axis(single_output_reshape, single_sorted_indices, axis=1).flatten()
        sorted_idx_global = np.take_along_axis(single_idx_reshape, single_sorted_indices, axis=1).flatten()
        if cols % blocksLen != 0:
            zeros_output = np.zeros(cols % blocksLen, dtype=sorted_output_global.dtype)
            zeros_index = np.zeros(cols % blocksLen, dtype=np.uint32)
            single_sorted_output_global = np.concatenate((sorted_output_global, zeros_output))
            single_sorted_idx_global = np.concatenate((sorted_idx_global, zeros_index))
            sorted_pairs_global = zip(single_sorted_output_global, single_sorted_idx_global)
        else:
            sorted_pairs_global = zip(sorted_output_global, sorted_idx_global)
        write(sorted_pairs_global, src_type)
    else:
        flat_input_group = np.concatenate(output_arr).flatten()
        flat_idx_group = np.concatenate(output_idx).flatten()
        sorted_indices_global = np.argsort(-flat_input_group, kind = 'stable',)
        sorted_output_global = flat_input_group[sorted_indices_global]
        sorted_idx_global = flat_idx_group[sorted_indices_global]
        zeros_output = np.zeros(input_num * cols - topk, dtype=sorted_output_global.dtype)
        zeros_index = np.zeros(input_num * cols - topk, dtype=np.uint32)
        topk_sorted_output_global = np.concatenate((sorted_output_global[:topk], zeros_output))
        topk_sorted_idx_global = np.concatenate((sorted_idx_global[:topk], zeros_index))
        
        if case_name.startswith("TMRGSORTTest.case_exhausted"):
            for i in range(input_num):
                zero_index = find_and_zero(topk_sorted_output_global, last_data[i])
                zero_after_index(topk_sorted_idx_global, zero_index)
        sorted_pairs_global = zip(topk_sorted_output_global, topk_sorted_idx_global)
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
    def __init__(self, case_name, datatype, row, src0_col, src1_col, src2_col, src3_col, input_num, topk, blockLen):
        self.case_name = case_name
        self.datatype = datatype
        self.row = row
        self.src0_col = src0_col
        self.src1_col = src1_col
        self.src2_col = src2_col
        self.src3_col = src3_col
        self.input_num = input_num
        self.topk = topk
        self.blockLen = blockLen

if __name__ == "__main__":
    case_params_list = [
        # col=128，表示64个数字+64个索引，实际内存大小是128 *sizeof(float)
        # TMRGSORTTest.case_multi, 多Tile
        tmrgsortParams("TMRGSORTTest.case_multi1", np.float32, 1, 128, 128, 128, 128, 4, 512, 0),
        tmrgsortParams("TMRGSORTTest.case_multi2", np.float16, 1, 128, 128, 128, 128, 4, 512, 0),
        tmrgsortParams("TMRGSORTTest.case_multi3", np.float32, 1, 128, 128, 128, 64, 4, 448, 0),
        tmrgsortParams("TMRGSORTTest.case_multi4", np.float32, 1, 128, 128, 64, 0, 3, 128, 0),
        # TMRGSORTTest.case_exhausted, 多Tile输入并开启耗尽模式
        tmrgsortParams("TMRGSORTTest.case_exhausted1", np.float32, 1, 64, 64, 0, 0, 2, 128, 0),
        tmrgsortParams("TMRGSORTTest.case_exhausted2", np.float16, 1, 256, 256, 256, 0, 3, 768, 0),
        # TMRGSORTTest.case_single, 单Tile
        tmrgsortParams("TMRGSORTTest.case_single1", np.float32, 1, 256, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single2", np.float32, 1, 320, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single3", np.float32, 1, 512, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single4", np.float32, 1, 640, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single5", np.float16, 1, 256, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single6", np.float16, 1, 320, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single7", np.float16, 1, 512, 0, 0, 0, 1, 0, 64),
        tmrgsortParams("TMRGSORTTest.case_single8", np.float16, 1, 1024, 0, 0, 0, 1, 0, 256),

        # TMRGSORTTest.case_topk，单Tile
        tmrgsortParams("TMRGSORTTest.case_topk1", np.float32, 1, 2048, 0, 0, 0, 1, 1024, 64),
        tmrgsortParams("TMRGSORTTest.case_topk2", np.float32, 1, 2048, 0, 0, 0, 1, 2048, 64),
        tmrgsortParams("TMRGSORTTest.case_topk3", np.float32, 1, 1280, 0, 0, 0, 1, 512, 64),
        tmrgsortParams("TMRGSORTTest.case_topk4", np.float16, 1, 2048, 0, 0, 0, 1, 1024, 64),
        tmrgsortParams("TMRGSORTTest.case_topk5", np.float16, 1, 2048, 0, 0, 0, 1, 2048, 64),
        tmrgsortParams("TMRGSORTTest.case_topk6", np.float16, 1, 1280, 0, 0, 0, 1, 512, 64),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)

        os.chdir(original_dir)


