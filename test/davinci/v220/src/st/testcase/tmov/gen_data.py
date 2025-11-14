#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import copy
import struct
# bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

class Dequantizer:
    def __init__(self):
        self.deqf16_factor = []
    
    def extract_quant_params(self, quant_value_uint64):
        """
        从uint64量化因子中提取参数
        64位结构:
        [63:36] - mcb + reserved (28位)
        [35:32] - N (4位，右移位数)
        [31:0]  - M1 (32位浮点数)
        M2 = M1 (不使用ReLU)
        """
        # 确保是整数类型
        quant_uint64 = int(quant_value_uint64)
        
        # 提取各个字段
        M1_uint32 = quant_uint64 & 0xFFFFFFFF  # [31:0] M1
        N_shift = (quant_uint64 >> 32) & 0xF   # [35:32] N (4位)
        mcb_and_reserved = (quant_uint64 >> 36) & 0xFFFFFFF  # [63:36] mcb + reserved
        
        mcb = mcb_and_reserved & 0x1
        
        # 将M1从uint32转换回float32
        M1_bytes = M1_uint32.to_bytes(4, byteorder='little', signed=False)
        M1 = np.frombuffer(M1_bytes, dtype=np.float32)[0]
        
        # 不使用ReLU，M2 = M1
        M2 = M1
        
        return mcb, N_shift, M1, M2
    
    def deqf16_quantization(self, quant_value_uint64, src_value_in_s32):
        """
        反量化函数 - 不使用ReLU
        """
        tmp0 = np.int32(src_value_in_s32)
        
        mcb, N, M1, M2 = self.extract_quant_params(quant_value_uint64)
        
        if mcb == 0:
            # 模式0: 直接32位转浮点
            tmp2 = tmp0.astype(np.float32)
        else:
            # 模式1: 右移N位 + 饱和到16位 + 转浮点
            # 使用算术右移保持符号
            if tmp0 < 0:
                shifted = -((-tmp0) >> N)  # 负数保持符号
            else:
                shifted = tmp0 >> N
                
            tmp1 = np.clip(shifted, -32768, 32767).astype(np.int16)
            tmp2 = tmp1.astype(np.float32)
        
        # 不使用ReLU，正负数都使用M1
        tmp3 = tmp2 * M1
        
        # 转half精度
        tmp4 = np.float16(tmp3)
        
        return tmp4
    
    def process_batch_column_based(self, quant_tensor, s32_elements):
        """
        列级量化处理 - 不使用ReLU
        """
        if len(s32_elements.shape) != 2:
            raise ValueError(f"输入应该是二维数组，当前形状: {s32_elements.shape}")
        
        M, N = s32_elements.shape
        
        # 确保quant_tensor是uint64类型
        if quant_tensor.dtype != np.uint64:
            quant_tensor = quant_tensor.astype(np.uint64)
        
        if len(quant_tensor) != N:
            raise ValueError(f"量化参数数量{len(quant_tensor)}与输出列数{N}不匹配")
        
        print(f"量化处理 (不使用ReLU):")
        print(f"  输入形状: ({M}, {N}), 范围: [{np.min(s32_elements)}, {np.max(s32_elements)}]")
        print(f"  量化参数: shape={quant_tensor.shape}")
        
        # 显示量化参数详情
        print(f"\n量化参数详情:")
        for col_idx in range(min(N, 3)):
            quant_value = quant_tensor[col_idx]
            mcb, N_shift, M1, M2 = self.extract_quant_params(quant_value)
            
            print(f"  第{col_idx}列:")
            print(f"    uint64值: 0x{quant_value:016X}")
            print(f"    解析: mcb={mcb}, N={N_shift}, M1={M1:.6f}, M2={M2:.6f}")
        
        # 处理每个元素
        results = np.zeros((M, N), dtype=np.float16)
        
        for row_idx in range(M):
            for col_idx in range(N):
                result = self.deqf16_quantization(quant_tensor[col_idx], s32_elements[row_idx, col_idx])
                results[row_idx, col_idx] = result
        
        print(f"  输出形状: {results.shape}, 范围: [{np.min(results):.6f}, {np.max(results):.6f}]")
        return results


def create_quant_tensor_uint64(nAlign):
    """
    创建uint64类型的量化参数张量 - 不使用ReLU
    """
    # 创建有意义的量化参数
    mcb_value = [0 for i in range(nAlign)]
    N_value = [1 for i in range(nAlign)]
    M1_value = [0.25 for i in range(nAlign)]  # 不同的缩放因子

    quant_tensor = np.zeros(nAlign, dtype=np.uint64)
    
    for i in range(nAlign):
        # 处理标量参数或列表参数
        if isinstance(mcb_value, (list, np.ndarray)):
            mcb = mcb_value[i] if i < len(mcb_value) else mcb_value[-1]
        else:
            mcb = mcb_value if mcb_value is not None else (i % 2)
        
        if isinstance(N_value, (list, np.ndarray)):
            N_shift = N_value[i] if i < len(N_value) else N_value[-1]
        else:
            N_shift = N_value if N_value is not None else ((i % 8) + 2)
        
        if isinstance(M1_value, (list, np.ndarray)):
            M1_val = M1_value[i] if i < len(M1_value) else M1_value[-1]
        else:
            M1_val = M1_value if M1_value is not None else (0.5 + (i % 10) * 0.1)
        
        # 限制N_shift在4位范围内
        N_shift = N_shift & 0xF
        
        # 不使用ReLU，reserved字段只包含mcb
        reserved = mcb  # 只有mcb位，没有ReLU模式位
        
        # 将M1从float32转换为uint32
        M1_array = np.array([M1_val], dtype=np.float32)
        M1_bytes = M1_array.tobytes()
        M1_uint32 = int.from_bytes(M1_bytes, byteorder='little', signed=False)
        
        # 构造64位uint64值
        quant_value = (reserved << 36) | (N_shift << 32) | M1_uint32
        quant_tensor[i] = quant_value
    
    return quant_tensor


def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def golden_data_saturation_convert(golden, l0c_type, temp_quant_tensor, M):
    # int8, >=127 to 127; <-128 to -128; uint8, >=255 to 255; < 0 to 0
    for i in range(M):
        golden[i, :] = golden[i, :] * temp_quant_tensor
    for i in range(len(golden)):
        for j in range(len(golden[0])):
            if l0c_type == np.int8:
                if golden[i][j] >= 127:
                    golden[i][j] = 127
                elif golden[i][j] <= -128:
                    golden[i][j] = -128
            elif l0c_type == np.float16:
                if golden[i][j] >= 255:
                    golden[i][j] = 255
                elif golden[i][j] < 0:
                    golden[i][j] = 0
    return golden

def gen_golden_data(case_name, param):
    src_type = param.atype
    l0c_type = param.ctype
    dst_type = param.gmtype
    bias_type = param.biastype

    m, k, n, start_m, start_k, is_bias, is_atrans, is_btrans =  param.m, param.k, param.n, param.start_m, param.start_k,True, param.is_atrans, param.is_btrans
    is_bias, is_quant, relu_mode, is_nd = param.is_bias, param.is_quant, param.relu_mode,param.is_nd
    
    biasNAlign = n
    scalingNAlign = n
    # bias需要64B对齐
    if bias_type == np.float16:
        biasNAlign = (int)((np.ceil((n * 2) / 64) * 64) / 2)
    elif bias_type == np.float32 or bias_type == np.int32:
        biasNAlign = (int)((np.ceil((n * 4) / 64) * 64) / 4)

    # fb需要128B对齐
    scalingNAlign = (int)((np.ceil((n * 8) / 128) * 128) / 8)


    x1_gm = np.random.randint(-1, 10, [m, k]).astype(src_type)
    x2_gm = np.random.randint(-1, 10, [k, n]).astype(src_type)
    bias_gm = np.random.randint(1, 10, [biasNAlign, ]).astype(bias_type)

    # 获取切片
    x1_slice = x1_gm[start_m:, start_k:]  # 从(rowIdx1, colIdx1)开始到结束
    x2_slice = x2_gm[start_k:, :]  # 从(rowIdx2, colIdx2)开始到结束

    # 根据实际维度调整索引方式
    # bias需要用户保证64B对齐，bias_gm中只有前n个数是有效值，剩余为无效值，只是为了满足对齐要求，不参与计算
    if bias_gm.ndim == 1:
        # 如果是一维数组，使用一维索引
        bias_slice = bias_gm[:n]  # 去掉第二个维度索引
    else:
        # 如果是二维数组，保持原代码
        bias_slice = bias_gm[:n, :]

    # A:[m-start_m, k-start_k]
    # B:[k-start_k, n]
    # C:[m-start_m, n]
    if is_bias:
        golden = (np.matmul(x1_slice.astype(l0c_type), x2_slice.astype(l0c_type)).astype(l0c_type) + bias_slice.astype(l0c_type)).astype(l0c_type)
    else:
        golden = (np.matmul(x1_slice.astype(l0c_type), x2_slice.astype(l0c_type)).astype(l0c_type)).astype(l0c_type)

    #-----------------------------------------量化------------------------------
    if dst_type == np.int8:
        temp_quant_tensor = np.random.randint(1, 5, [scalingNAlign, ]).astype(np.float32)
        temp_quant_tensor_slice = temp_quant_tensor[:n]
        # scaling需要用户保证128B对齐，scaling_gm中只有前n个数是有效值，剩余为无效值，只是为了满足对齐要求，不参与计算
        temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
        for i, _ in enumerate(temp_quant_tensor_api):
            #将每个float32的位模式转换为uint64，保持浮点数的位模式
            temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor[i]))[0]
            # 对于B8输出场景要在数据中设置特性标志位，表示输出是u8还是s8:[46]=0 , dst_type=uint8; [46]=1 ,dst_type=int8
            if dst_type == np.int8:
                temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)

        scaling_gm = np.frombuffer(temp_quant_tensor_api, np.uint64)
        if is_quant:
            golden = golden_data_saturation_convert(golden, dst_type, temp_quant_tensor_slice, m).astype(dst_type)
    elif dst_type == np.float16:
        # 创建反量化器
        dequantizer = Dequantizer()
        scaling_gm = create_quant_tensor_uint64(scalingNAlign)
        # scaling需要用户保证128B对齐，scaling_gm中只有前n个数是有效值，剩余为无效值，只是为了满足对齐要求，不参与计算
        scaling_gm_slice = scaling_gm[:n]
        if is_quant:
            golden = dequantizer.process_batch_column_based(scaling_gm_slice, golden)
    else:
        scaling_gm = np.random.randint(1, 5, [scalingNAlign, ]).astype(np.uint64)
    #- -----------------------------------------------------------------------

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()#[N,K]

    c0_size = 16
    if src_type == np.float32:
        c0_size = 8
    elif src_type == np.int8:
        c0_size = 32

    #转成NZ格式的输入
    if not is_nd:
        x1_gm = x1_gm.reshape((int(x1_gm.shape[0] / 16), 16, int(x1_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
        x1_gm = x1_gm.reshape(x1_gm.shape[0] * x1_gm.shape[1], x1_gm.shape[2] * x1_gm.shape[3])

        x2_gm = x2_gm.reshape((int(x2_gm.shape[0] / 16), 16, int(x2_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
        x2_gm = x2_gm.reshape(x2_gm.shape[0] * x2_gm.shape[1], x2_gm.shape[2] * x2_gm.shape[3])

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    scaling_gm.tofile("./scaling_gm.bin")
    golden.tofile("./golden.bin")

    os.chdir(original_dir)

class tmovParams:
    def __init__(self, atype, btype, ctype, biastype, gmtype, m, n, k, is_atrans=0, is_btrans=0, is_bias=0, is_quant=0, relu_mode=0, is_nd = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.biastype = biastype
        self.gmtype = gmtype
        self.m = m
        self.n = n
        self.k = k
        self.start_m = 0
        self.start_k = 0
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.is_bias = is_bias
        self.is_quant = is_quant
        self.relu_mode = relu_mode
        self.is_nd = is_nd

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        # copy_l1_bias
        #TMOVTest.caseName_model_inputType_biasType_isAtranspose_isBtranspose_isBias_isQuant_reluMode
        "TMOVTest.case1_bias_static_half_float_0_1_1_0_0_param", # 此名称需要和 TEST_F(TMATMULTest, case1)定义的名称一致
        "TMOVTest.case2_bias_static_int8_int32_0_1_1_0_0_param",
        "TMOVTest.case3_bias_static_float_float_0_1_1_0_0_param",
        "TMOVTest.case4_bias_dynamic_half_half_0_1_1_0_0_param",
        "TMOVTest.case5_bias_dynamic_float_half_0_1_1_0_0_param",
        "TMOVTest.case6_bias_static_float_half_0_1_1_0_0_param",
        # copy_l1_fb
        # #TMOVTest.caseName_model_l0cType_gmType_isAtranspose_isBtranspose_isBias_isQuant_reluMode
        "TMOVTest.case11_scaling_static_int32_int8_0_1_0_1_0_param",
        "TMOVTest.case12_scaling_static_int32_half_0_1_0_1_0_param",
        "TMOVTest.case13_scaling_static_float_int8_0_1_0_1_0_param",
        # copy_l1_bias + copy_l1_fb + ndInput + dynamic + unalign
        "TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param",
    ]

    case_params_list = [
        #tmovParams(atype, btype, l0ctype, biastype, gmtype, m, n, k, is_atrans=0, is_btrans=0, is_bias = 0, is_quant=0, relu_mode=0, is_nd=0)
        # copy_l1_bias
        tmovParams(np.float16, np.float16, np.float32, np.float32, np.float32, 64, 32, 80,  0, 1, 1, 0, 0),#case1
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int32, 128, 64, 128, 0, 1, 1, 0, 0),#case2
        tmovParams(np.float32, np.float32,  np.float32, np.float32, np.float32, 128, 48, 64,  0, 1, 1, 0, 0),#case3
        tmovParams(np.float16, np.float16, np.float32, np.float16, np.float32, 64, 32, 80,  0, 1, 1, 0, 0),#case4
        tmovParams(np.float32, np.float32,  np.float32, np.float16, np.float32, 112, 48, 96,  0, 1, 1, 0, 0),#case5
        tmovParams(np.float32, np.float32,  np.float32, np.float16, np.float32, 64, 128, 96,  0, 1, 1, 0, 0),#case6
        # copy_l1_fb
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int8, 128, 112, 32,  0, 1, 0, 1, 0),#case11
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.float16, 144, 80, 160,  0, 1, 0, 1, 0),#case12
        tmovParams(np.float16, np.float16, np.float32, np.float32, np.int8, 64, 32, 80, 0, 1, 0, 1, 0),#case13
        # copy_l1_bias + copy_l1_fb + ndInput + dynamic + unalign
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int8, 60, 17, 80, 0, 1, 1, 1, 0, 1),#case14
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)