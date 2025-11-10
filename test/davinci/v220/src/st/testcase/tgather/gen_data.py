#!/user/bin/python3
# coding=utf-8

import os

import numpy as np
np.random.seed(19)


# 生成数据  p0101 = 1, // 1: 01010101...0101 # 每个repeat内每两个元素取第一个元素
# 生成数据  p1010 = 2, // 2: 10101010...1010 # 每个repeat内每两个元素取第二个元素
# 生成数据  p0001 = 3, // 3: 00010001...0001 # 每个repeat内每四个元素取第一个元素
# 生成数据  p0010 = 4, // 4: 00100010...0010 # 每个repeat内每四个元素取第二个元素
# 生成数据  p0100 = 5, // 5: 01000100...0100 # 每个repeat内每四个元素取第三个元素
# 生成数据  p1000 = 6, // 6: 10001000...1000 # 每个repeat内每四个元素取第四个元素
# 生成数据  p1111 = 7, // 7: 11111111...1111 # 每个repeat内取全部元素
P0101 = 1
P1010 = 2
P0001 = 3
P0010 = 4
P0100 = 5
P1000 = 6
P1111 = 7

# 需要合tgather_common.h里的对应一致
HALF_P0101_ROW = 5
HALF_P0101_COL = 128
HALF_P1010_ROW = 7
HALF_P1010_COL = 1024
HALF_P0001_ROW = 3
HALF_P0001_COL = 1056
HALF_P0010_ROW = 4
HALF_P0010_COL = 128
HALF_P0100_ROW = 5
HALF_P0100_COL = 256
HALF_P1000_ROW = 6
HALF_P1000_COL = 288
HALF_P1111_ROW = 7
HALF_P1111_COL = 320

FLOAT_P0101_ROW = 4
FLOAT_P0101_COL = 64
FLOAT_P1010_ROW = 7
FLOAT_P1010_COL = 1024
FLOAT_P0001_ROW = 3
FLOAT_P0001_COL = 1056
FLOAT_P0010_ROW = 4
FLOAT_P0010_COL = 128
FLOAT_P0100_ROW = 5
FLOAT_P0100_COL = 256
FLOAT_P1000_ROW = 6
FLOAT_P1000_COL = 288
FLOAT_P1111_ROW = 7
FLOAT_P1111_COL = 320


class TGatherParamsBase:
    def __init__(self, name):
        self.testName = name

class TGatherParamsMasked(TGatherParamsBase):
    def __init__(self, name, dstType, srcType, row, col, pattern):
        super().__init__(name)
        self.dstType = dstType
        self.srcType = srcType
        self.row = row
        self.col = col
        self.pattern = pattern

class TGatherParams1D(TGatherParamsBase):
    def __init__(self, name, srcType, srcRow, srcCol, dstRow, dstCol):
        super().__init__(name)
        self.srcType = srcType
        self.srcRow = srcRow
        self.srcCol = srcCol
        self.dstRow = dstRow
        self.dstCol = dstCol

def Gather1D(src, indices):
    output = np.zeros_like(indices, dtype=src.dtype)
    for i in range(indices.shape[0]):
        output[i] = src[indices[i]]
    return output

def gen_golden_data(param: TGatherParamsBase):
    if isinstance(param, TGatherParamsMasked):
        src_type = param.srcType
        dst_type = param.dstType
        row = param.row
        col = param.col
        pattern = param.pattern
        x1_gm = np.random.randint(1, 100, [row, col]).astype(src_type)
        x1_gm.tofile("./x1_gm.bin")
        res = np.zeros((row, col))
        if pattern == P0101 :
            res = x1_gm[:, 0::2]
        elif pattern == P1010 :
            res = x1_gm[:, 1::2]
        elif pattern == P0001 :
            res = x1_gm[:, 0::4]
        elif pattern == P0010 :
            res = x1_gm[:, 1::4]
        elif pattern == P0100 :
            res = x1_gm[:, 2::4]
        elif pattern == P1000 :
            res = x1_gm[:, 3::4]
        elif pattern == P1111 :
            res = x1_gm[:, :]
        
        if pattern == 255 and src_type == np.half:
            newarray = x1_gm.reshape(row, col // 4, 4)
            selected = newarray[:, :, 2:4]
            res = selected.reshape(-1)

        res_flat = res.flatten()
        pad_length = max(0, row*col - len(res_flat))
        pad_res = np.pad(res_flat, (0, pad_length), 'constant', constant_values=0)
        golden = pad_res.reshape(row, col)

        x1_gm.tofile("./x1_gm.bin")
        golden.tofile("./golden.bin")
        os.chdir(original_dir)
    elif isinstance(param, TGatherParams1D):
        output = np.zeros([param.dstRow*param.dstCol]).astype(param.srcType)
        src_data = np.random.randint(-20, 20, (param.srcRow*param.srcCol)).astype(param.srcType)
        src_data.tofile("./src0.bin")
        indices = np.random.randint(0, param.srcRow*param.srcCol, (param.dstRow*param.dstCol)).astype(np.int32)
        indices.tofile("./src1.bin")
        golden = Gather1D(src_data, indices)
        golden.tofile("./golden.bin")
        pass

class tgatherParams:
    def __init__(self, dsttype, srctype, row, col, pattern):
        self.dsttype = dsttype
        self.srctype = srctype
        self.row = row
        self.col = col
        self.pattern = pattern

if __name__ == "__main__":
    case_params_list = [
        TGatherParamsMasked("TGATHERTest.case1_float_P0101", np.float32, np.float32, FLOAT_P0101_ROW, FLOAT_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_float_P1010", np.float32, np.float32, FLOAT_P1010_ROW, FLOAT_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_float_P0001", np.float32, np.float32, FLOAT_P0001_ROW, FLOAT_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_float_P0010", np.float32, np.float32, FLOAT_P0010_ROW, FLOAT_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_float_P0100", np.float32, np.float32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_float_P1000", np.float32, np.float32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_float_P1111", np.float32, np.float32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
        TGatherParamsMasked("TGATHERTest.case1_half_P0101", np.half, np.half, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_half_P1010", np.half, np.half, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_half_P0001", np.half, np.half, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_half_P0010", np.half, np.half, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_half_P0100", np.half, np.half, HALF_P0100_ROW, HALF_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_half_P1000", np.half, np.half, HALF_P1000_ROW, HALF_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_half_P1111", np.half, np.half, HALF_P1111_ROW, HALF_P1111_COL, P1111),

        TGatherParamsMasked("TGATHERTest.case1_U16_P0101", np.uint16, np.uint16, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_U16_P1010", np.uint16, np.uint16, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_I16_P0001", np.int16, np.int16, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_I16_P0010", np.int16, np.int16, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_U32_P0100", np.uint32, np.uint32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_I32_P1000", np.int32, np.int32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_I32_P1111", np.int32, np.int32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),

        # Test cases for Tgather1D
        # TGatherParams1D("TGATHERTest.TestName", datatype, srcrow, srccol, dstRow, dstCol)
        TGatherParams1D("TGATHERTest.case_1D_float_32x1024_16x64", np.float32, 32, 1024, 16, 64),
        TGatherParams1D("TGATHERTest.case_1D_int32_32x512_16x256", np.int32, 32, 512, 16, 256),
        TGatherParams1D("TGATHERTest.case_1D_half_16x1024_16x128", np.float16, 16, 1024, 16, 128),
        TGatherParams1D("TGATHERTest.case_1D_int16_32x256_32x64", np.int16, 32, 256, 32, 64),
    ]

    for i, case in enumerate(case_params_list):
        if not os.path.exists(case.testName):
            os.makedirs(case.testName)
        original_dir = os.getcwd()
        os.chdir(case.testName)
        gen_golden_data(case)
        os.chdir(original_dir)