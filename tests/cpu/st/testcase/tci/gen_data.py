#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np


NUM_ELEMS = 16
START_S0 = 0
START_S100 = 100


def write_case(_case_name: str, start: int, is_descending: bool):
    if is_descending:
        golden = (start - np.arange(NUM_ELEMS, dtype=np.int32)).astype(np.int32)
    else:
        golden = (start + np.arange(NUM_ELEMS, dtype=np.int32)).astype(np.int32)
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "testcases"), exist_ok=True)

    cases = [
        ("TCI_Test.case_i32_asc_S0", START_S0, False),
        ("TCI_Test.case_i32_desc_S100", START_S100, True),
    ]

    cwd = os.getcwd()
    for name, start, is_descending in cases:
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        write_case(name, start, is_descending)
        os.chdir(cwd)
