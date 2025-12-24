#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

SHORT=r:,v:,c:,a:,i
LONG=run-mode:,soc-version:,case:,cases:,intermediate
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-c | --case )
            CASE_FILTER="$2"
            shift 2;;
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (-i | --intermediate )
            INTERMEDIATE=1
            shift 1;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

GEN_CASE_ARGS=()
# Handle missing value after -c/--case (e.g. user passed -c and then --cases)
if [[ -n "${CASE_FILTER:-}" && "${CASE_FILTER}" == --* ]]; then
    CASE_FILTER=""
fi

if [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_ENTRIES <<< "${CASES_RAW}"
    for entry in "${CASE_ENTRIES[@]}"; do
        GEN_CASE_ARGS+=(--cases "$entry")
    done
elif [[ -n "${CASE_FILTER:-}" ]]; then
    # If only a single case filter was provided, ensure generation for numeric tuple filters
    if [[ "${CASE_FILTER}" == *","* ]]; then
        GEN_CASE_ARGS+=(--cases "${CASE_FILTER}")
    fi
fi

echo "[RUN.SH] CASE_FILTER=${CASE_FILTER:-}<none>"
echo "[RUN.SH] CASES_RAW=${CASES_RAW:-}<none>"
echo "[RUN.SH] GEN_CASE_ARGS=${GEN_CASE_ARGS[*]:-<none>}"
echo "[RUN.SH] INTERMEDIATE=${INTERMEDIATE:-0}"

python3 ../scripts/generate_cases.py "${GEN_CASE_ARGS[@]}"

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} ..
make -j16

EXTRA_BIN_ARGS=()
if [[ -n "${INTERMEDIATE:-}" ]]; then
    EXTRA_BIN_ARGS+=(--intermediate)
fi

if [[ -n "${CASE_FILTER:-}" ]]; then
    python3 ../scripts/gen_data.py --case="${CASE_FILTER}" "${GEN_CASE_ARGS[@]}"
    time ./fa_performance --case="${CASE_FILTER}" "${EXTRA_BIN_ARGS[@]}"
elif [[ -n "${CASES_RAW:-}" ]]; then
    python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}"
    time ./fa_performance --cases="${CASES_RAW}" "${EXTRA_BIN_ARGS[@]}"
else
    python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}"
    time ./fa_performance "${EXTRA_BIN_ARGS[@]}"
fi
