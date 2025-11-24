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
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

#print usage message
usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help  Print usage"
  echo "    --pkg Build run package"
  echo "    --run_all run all st on sim"
  echo "    --run_simple run some st on board"
  echo ""
}

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
  ENABLE_PACKAGE=FALSE
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  PLATFORM_MODE=""
  INST_NAME=""

  parsed_args=$(getopt -a -o j:hvuO: -l help,verbose,cov,make_clean,noexec,pkg,run_all,run_simple,cann_3rd_lib_path: -- "$@") || {
  usage
  exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --run_all)
        ENABLE_BUILD_ALL=TRUE
        shift
        ;;
      --run_simple)
        ENABLE_SIMPLE_ST=TRUE
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done
}

run_simple_st() {
  echo $dotted_line
  echo "Start to run simple st"
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

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}
  fi
}

build_package() {
  echo "---------------package start-----------------"
  clean_build_out
  mkdir $BUILD_PATH
  mkdir $BUILD_OUT_PATH
  cd $BUILD_PATH
  cmake ..
  make package
  echo "---------------package end-----------------"
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
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    build_package
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
