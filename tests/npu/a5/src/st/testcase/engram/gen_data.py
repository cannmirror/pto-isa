#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import sys
import numpy as np

NUM_HEADS = 8
GATE_BIAS = np.float64(0.125)

TEST_CASES = [
    ("baseline_float_128x128_8x128", 128, 128, 8, 128),
    ("baseline_float_256x256_8x256", 256, 256, 8, 256),
    ("baseline_float_512x512_8x512", 512, 512, 8, 512),
    ("baseline_float_1024x1024_8x1024", 1024, 1024, 8, 1024),
    ("fused_float_128x128_8x128", 128, 128, 8, 128),
    ("fused_float_256x256_8x256", 256, 256, 8, 256),
    ("fused_float_512x512_8x512", 512, 512, 8, 512),
    ("fused_float_1024x1024_8x1024", 1024, 1024, 8, 1024),
]


def generate_golden(test_name, table_rows, table_cols, num_heads, emb_dim, out_dir):
    """Generate input + golden data for one test case."""
    os.makedirs(out_dir, exist_ok=True)

    table = np.zeros((table_rows, table_cols), dtype=np.float64)
    for r in range(table_rows):
        for c in range(table_cols):
            table[r, c] = ((r + c) % 8 + 1) * 0.0625

    gate_weight = np.zeros(emb_dim, dtype=np.float64)
    for j in range(emb_dim):
        gate_weight[j] = 0.001953125 * (1 + j % 4)

    indices = np.array([i % table_rows for i in range(num_heads)], dtype=np.int32)

    hidden = np.full(emb_dim, 0.25, dtype=np.float64)

    # Stage 1: Gather
    gathered = np.zeros((num_heads, emb_dim), dtype=np.float64)
    for h in range(num_heads):
        gathered[h, :] = table[int(indices[h]), :emb_dim]

    # Stage 2: Aggregate (column-wise mean)
    agg = gathered.mean(axis=0)

    # Stage 3: Context Gating
    dot = np.sum(hidden * gate_weight) + GATE_BIAS
    gate_score = 1.0 / (1.0 + np.exp(-dot))
    golden = hidden + gate_score * agg

    table.astype(np.float32).tofile(os.path.join(out_dir, "table.bin"))
    indices.tofile(os.path.join(out_dir, "indices.bin"))
    hidden.astype(np.float32).tofile(os.path.join(out_dir, "hidden.bin"))
    gate_weight.astype(np.float32).tofile(os.path.join(out_dir, "gate_weight.bin"))
    golden.astype(np.float32).tofile(os.path.join(out_dir, "golden.bin"))

    print(f"  [{test_name}] table={table_rows}x{table_cols} heads={num_heads} emb={emb_dim}")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    filter_name = sys.argv[2] if len(sys.argv) > 2 else None

    print("Engram Golden Data Generator")
    print(f"Output: {base_dir}\n")

    for test_name, table_rows, table_cols, num_heads, emb_dim in TEST_CASES:
        if filter_name and filter_name != test_name:
            continue
        case_dir = os.path.join(base_dir, f"ENGRAMTest.{test_name}")
        generate_golden(test_name, table_rows, table_cols, num_heads, emb_dim, case_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
