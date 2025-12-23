#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${SCRIPT_DIR}/../build"

set -x
python3 "${SCRIPT_DIR}/../scripts/pipeline_log_analysis.py" \
	--device-addrs "${build}/device_addrs.toml" \
	--cube-start "${build}/core0.cubecore0.instr_popped_log.dump" \
	--cube-end "${build}/core0.cubecore0.instr_log.dump" \
	--vec-start "${build}/core0.veccore0.instr_popped_log.dump" \
	--vec-end "${build}/core0.veccore0.instr_log.dump" \
	--out-csv timeline.csv \
	--out-json timeline.json \
	--out-agg timeline_agg.csv \
	--out-svg timeline.svg
