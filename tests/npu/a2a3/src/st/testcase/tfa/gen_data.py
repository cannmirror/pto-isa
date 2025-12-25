#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
Generate Q/K input and golden output for TBMM_QK 128x128x128 (float32)
Writes: q.bin, k.bin, golden.bin
"""
import os
import shutil
import numpy as np

np.random.seed(7)

S0_BASE = 64
HEAD_SIZE = 128

def gen_case(path, s0, s1, head_size=HEAD_SIZE):
    # generate inputs in FP16, compute golden in FP32
    q_fp32 = (np.random.randn(s0, head_size).astype(np.float16) * 1.5).astype(np.float32)
    k_fp32 = (np.random.randn(head_size, s1).astype(np.float16) * 1.5).astype(np.float32)
    q = q_fp32.astype(np.float16)
    k = k_fp32.astype(np.float16)
    golden = (q.astype(np.float32).dot(k.astype(np.float32))).astype(np.float32)

    # write FP16 inputs and FP32 golden
    q.tofile(os.path.join(path, 'q.bin'))
    k.tofile(os.path.join(path, 'k.bin'))
    kt = k.T.astype(np.float16)    
    kt.tofile(os.path.join(path, 'kt.bin'))       
    golden.tofile(os.path.join(path, 'golden.bin'))
    # also produce softmax x_exp (per-row) saved as FP16 and tmp_float_exp saved as FP32
    # compute softmax in tiled fashion by Cube_S1 tiles (tile size 128)
    arr_f32 = golden.astype(np.float32)
    scale = 1/np.sqrt(HEAD_SIZE)
    Cube_S1 = 128
    num_tiles = s1 // Cube_S1

    # allocate full arrays to collect per-tile exponentials and per-tile global sums
    full_exp = np.zeros((s0, s1), dtype=np.float32)
    global_sums = []
    exp_max_parts = []

    # emulate TSOFTMAXFA recurrence across tiles to compute new_global_sum and exp_max per tile
    global_max = None
    global_sum = None

    for ti in range(num_tiles):
        c0 = ti * Cube_S1
        c1 = c0 + Cube_S1
        tile = arr_f32[:, c0:c1]
        # local max per row for this tile
        local_max = np.max(tile, axis=1, keepdims=True).astype(np.float32)
        if global_max is not None:
            local_max = np.maximum(local_max, global_max).astype(np.float32) 
        if ti == 0:
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = (np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32))
            exp_max_tile = np.ones_like(new_global_max).astype(np.float32)
        else:
            # exp_max = exp((global_max - local_max) * scale)
            exp_max = (global_max - local_max).astype(np.float32)
            exp_max = np.exp(exp_max * scale).astype(np.float32)
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = exp_max * global_sum + (np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32) )
            exp_max_tile = exp_max

        # record results and update global state
        full_exp[:, c0:c1] = tmp_float_exp
        global_sums.append(new_global_sum.reshape(-1))
        exp_max_parts.append(exp_max_tile.reshape(-1))
        global_max = new_global_max
        global_sum = new_global_sum

    tmp_float_exp = full_exp
    # p saved as FP16 (store raw exponentials per tile as half)
    soft = (full_exp).astype(np.float16)
    soft.tofile(os.path.join(path, 'p.bin'))
    tmp_float_exp.tofile(os.path.join(path, 'p_fp32.bin'))

    # generate random V (S1 x HEAD_SIZE) and compute y = soft (S0 x S1) dot V (S1 x HEAD_SIZE)
    v_fp32 = (np.random.randn(s1, head_size).astype(np.float16) * 1.2).astype(np.float32)
    v = v_fp32.astype(np.float16)
    # soft (S0 x S1) as float32
    soft_f32 = soft.astype(np.float32)
    # compute full pv by accumulating per-tile partials
    pv = np.zeros((s0, head_size), dtype=np.float32)
    # compute per-tile partials (Cube_S1=128)
    num_tiles = s1 // 128
    pv_parts = []
    for ti in range(num_tiles):
        c0 = ti * 128
        soft_tile = soft_f32[:, c0:c0+128]
        v_tile = v[c0:c0+128, :].astype(np.float32)
        pv_part = (soft_tile.dot(v_tile)).astype(np.float32)
        pv_parts.append(pv_part)
        pv += pv_part

    v.tofile(os.path.join(path, 'v.bin'))
    vt = v.T.astype(np.float16)    
    vt.tofile(os.path.join(path, 'vt.bin'))       
    pv.tofile(os.path.join(path, 'pv.bin'))
    # write per-tile partials as pv_part0.bin, pv_part1.bin
    for idx, part in enumerate(pv_parts):
        part.tofile(os.path.join(path, f'pv_part{idx}.bin'))
    # write per-tile global_sum and exp_max parts
    for idx, g in enumerate(global_sums):
        g.astype(np.float32).tofile(os.path.join(path, f'global_sum_part{idx}.bin'))
    for idx, e in enumerate(exp_max_parts):
        e.astype(np.float32).tofile(os.path.join(path, f'exp_max_part{idx}.bin'))

    # compute running output o: use exp_max per-tile for accumulation and divide by new_global_sum on last tile
    o_running = np.zeros((s0, head_size), dtype=np.float32)
    for ti, part in enumerate(pv_parts):
        if ti == 0:
            o_running = part.copy()
        else:
            exp_max_tile = exp_max_parts[ti].reshape((s0, 1)).astype(np.float32)
            o_running = exp_max_tile * o_running + part
            if ti == num_tiles - 1:
                new_global_sum_tile = global_sums[ti].reshape((s0, 1)).astype(np.float32)
                o_running = o_running / new_global_sum_tile
        # write per-iteration o
        o_running.astype(np.float32).tofile(os.path.join(path, f'o_part{ti}.bin'))
    # write final running output
    o_running.astype(np.float32).tofile(os.path.join(path, 'o.bin'))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cases = [
        ('TFATest.case_float_H_128_S0_64_S1_256', (S0_BASE, HEAD_SIZE, 256)),
        ('TFATest.case_float_H_128_S0_64_S1_128', (S0_BASE, HEAD_SIZE, 128)),
        ('TFATest.case_float_H_128_S0_64_S1_512', (S0_BASE, HEAD_SIZE, 512)),
        ('TFATest.case_float_H_128_S0_128_S1_512', (128, HEAD_SIZE, 512)),
        ('TFATest.case_float_H_128_S0_128_S1_2048', (128, HEAD_SIZE, 2048)),
        # ('TFATest.case_float_H_128_S0_128_S1_8192', (128, HEAD_SIZE, 8192)),
    ]
    for name, (s0, head_size, s1) in cases:
        case_dir = os.path.join(script_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        gen_case(case_dir, s0, s1, head_size)
        # Also produce a corresponding debug directory with suffix
        # "_precision_debug" so tests that expect intermediate files
        # can read from there as well. This applies for all S1 values
        # (including S1==512).
        dbg_name = name + '_precision_debug'
        dbg_dir = os.path.join(script_dir, dbg_name)
        os.makedirs(dbg_dir, exist_ok=True)
        for fname in os.listdir(case_dir):
            src = os.path.join(case_dir, fname)
            dst = os.path.join(dbg_dir, fname)
            shutil.copy(src, dst)
