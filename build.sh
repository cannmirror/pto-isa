#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# ============================================================================

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ASCEND_ENV_PATH="${ASCEND_HOME_PATH}/bin"

print_success() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

checkopts() {
  ENABLE_SIMPLE_ST=FALSE
  ENABLE_BUILD_ALL=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  PLATFORM_MODE=""
  INST_NAME=""

  if [[ "$1" == "--run_example" ]]; then
    ENABLE_RUN_EXAMPLE=TRUE
    INST_NAME="$2"
    EXAMPLE_NAME="$3"
    EXAMPLE_MODE="$4"
    PLATFORM_MODE="$5"
  elif [ "$1" == "--run_all" ]; then
    ENABLE_BUILD_ALL=TRUE
  elif [ "$1" == "--run_simple" ]; then
    ENABLE_SIMPLE_ST=TRUE
  fi
}

run_simple_st() {
  echo $dotted_line
  echo "Start to run simple st"
  cat /usr/local/Ascend/latest/compiler/version.info
  chmod +x run_st.sh
  ./run_st.sh run_simple
  echo "execute samples success"
}

run_all_st() {
  echo $dotted_line
  echo "Start to run all st"
  chmod +x run_st.sh
  ./run_st.sh dailyBuild
  echo "execute samples success"
}

run_example() {
  echo $dotted_line
  echo "Start to run example"
  python3 test/script/run_st.py -r $PLATFORM_MODE -v $EXAMPLE_MODE -t $INST_NAME -g $$EXAMPLE_NAME
  echo "execute samples success"
}

main() {
  checkopts "$@"
  ulimit -n 65535
  if [ "$ENABLE_SIMPLE_ST" == "TRUE" ]; then
      run_simple_st
  fi
  if [ "$ENABLE_BUILD_ALL" == "TRUE" ]; then
      run_all_st
  fi
  if [ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]; then
      run_example
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
