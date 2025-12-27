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


def matmul_reference(a, b):
    # Avoid BLAS calls; use pure NumPy elementwise/broadcasting.
    # a: (m, k), b: (k, n) -> (m, n)
    return (a[:, :, None] * b[None, :, :]).sum(axis=1)


def softmax_rows(x):
    # Stable softmax along last dim.
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    denom = np.sum(ex, axis=1, keepdims=True)
    return ex / denom


def gen_case(out_dir, seq_len, head_dim, seed, init_scale):
    rng = np.random.default_rng(seed)

    q = (rng.standard_normal((seq_len, head_dim), dtype=np.float32) * init_scale).astype(np.float32)
    k = (rng.standard_normal((seq_len, head_dim), dtype=np.float32) * init_scale).astype(np.float32)
    v = (rng.standard_normal((seq_len, head_dim), dtype=np.float32) * init_scale).astype(np.float32)

    scale = np.float64(1.0 / np.sqrt(float(head_dim)))
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)

    scores = matmul_reference(q64, k64.T) * scale
    probs = softmax_rows(scores)
    out = matmul_reference(probs, v64).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    q.tofile(os.path.join(out_dir, "x1_gm.bin"))
    k.tofile(os.path.join(out_dir, "x2_gm.bin"))
    v.tofile(os.path.join(out_dir, "x3_gm.bin"))
    out.tofile(os.path.join(out_dir, "golden.bin"))


if __name__ == "__main__":
    seq_len = 64
    head_dim = 32
    seed = 20251220
    init_scale = 0.02

    cases = [
        "TFLASHATTNTest.case1",
    ]

    for case in cases:
        gen_case(case, seq_len=seq_len, head_dim=head_dim, seed=seed, init_scale=init_scale)

