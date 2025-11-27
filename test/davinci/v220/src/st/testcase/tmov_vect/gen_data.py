import os
import numpy as np 

np.random.seed(0)


def get_golden_data(case_name, param):
    data_type = param.data_type
    rows = param.src_valid_row
    cols = param.src_valid_col
    input_arr = np.random.uniform(0, 20, size=(rows, cols)).astype(data_type) 
    input_arr.tofile("input_arr.bin")
    golden = input_arr.copy()
    golden.tofile(f"golden.bin")


class tmovParams:
    def __init__(self, data_type, src_valid_row, src_valid_col, dst_valid_row, dst_valid_col):
        self.data_type = data_type
        self.src_valid_row = src_valid_row
        self.src_valid_col = src_valid_col
        self.dst_valid_row = dst_valid_row
        self.dst_valid_col = dst_valid_col

    
if __name__ == "__main__":
    case_name_list = [f"TMOVTest.vect_copy_case{i}" for i in range(1, 16)]

    case_name_list = [
        tmovParams(np.float32, 64, 64, 64, 64),
        tmovParams(np.float32, 32, 32, 32, 32),
        tmovParams(np.float32, 128, 128, 128, 128),
        tmovParams(np.float32, 128, 32, 128, 32),
        tmovParams(np.float32, 128, 64, 128, 64),

        tmovParams(np.float16, 64, 64, 64, 64),
        tmovParams(np.float16, 32, 32, 32, 32),
        tmovParams(np.float16, 128, 128, 128, 128),
        tmovParams(np.float16, 128, 32, 128, 32),
        tmovParams(np.float16, 128, 64, 128, 64),
        
        tmovParams(np.uint8, 64, 64, 64, 64),
        tmovParams(np.uint8, 32, 32, 32, 32),
        tmovParams(np.uint8, 128, 128, 128, 128),
        tmovParams(np.uint8, 128, 32, 128, 32),
        tmovParams(np.uint8, 128, 64, 128, 64)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)

        original_dir = os.getcwd()
        os.chdir(case_name)

        get_golden_data(case_name, case_name_list[i])

        os.chdir(original_dir)
    