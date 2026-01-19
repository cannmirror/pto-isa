#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOOLKIT_INSTALL_PATH=""
TOOLKIT_PACKAGE_PATH=""
DOWNLOAD_DIR="./downloads"
PTO_ISA_PKG=""
GTEST_PKG=""
LOG_FILE="./setup_pto_isa_env.log"

CURRENT_DATE=$(date '+%Y%m%d')
CURRENT_YEAR=$(date '+%Y')
CURRENT_MONTH=$(date '+%m')
CURRENT_DAY=$(date + '%d')

MAX_BACKTRACK_DAYS=3

log() {
  local level="$1"
  local message="$2"
  echo -e "${level}[$(date '+%Y-%m-%d %H:%M:%S')] $message${NC}" | tee -a "$LOG_FILE"
}

info() { log "${GREEN}[INFO]${NC}" "$1"; }
warn() { log "${YELLOW}[WARN]${NC}" "$1"; }
error() { log "${RED}[ERROR]${NC}" "$1"; }

detect_arch() {
  local arch=$(uname -m)
  case $arch in
    aarch64|arm64)
      echo "aarch64"
      ;;
    x86_64|amd64)
      echo "x86_64"
      ;;
    *)
      error "not support arch: $arch"
      exit 1
      ;;
  esac
}

check_toolkit_installed() {
  local path="$1"
  local setenv_path="${path}/cann/bin/setenv.bash"

  if [[ -f "$setenv_path" ]]; then
    info "toolkit installed: $path"
    return 0
  else
    warn "toolkit has not installed"
    return 1
  fi
}

install_toolkit() {
  local package_path="$1"
  local install_path="$2"

  if [[ ! -f "$package_path" ]]; then
    error "toolkit package not exist: $package_path"
    exit 1
  fi
  info "install toolkit package to $install_path"

  if [[ "$package_path" == *.run ]]; then
    chmod +x "$package_path"
    "$package_path" --full --quiet --install-path="$install_path"
  else
    error "not support format: $package_path"
    exit 1
  fi

  if [[ $? -ne 0 ]]; then
    error "install toolkit failed!"
    exit 1
  fi

  info "install toolkit success!"
}

download_pto_isa_run() {
  local arch="$1"
  local base_url="http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/${CURRENT_YEAR}${CURRENT_MONTH}"

  local try_dates=()
  local current=$(date -d "$CURRENT_DATE" +%s)

  for i in $(seq 0 $((MAX_BACKTRACK_DAYS - 1))); do
    local target_date=$(date -d "@$((current - i * 86400))" '+%Y%m%d')
    try_dates+=("$target_date")
  done

  local filename="cann-pto-isa_8.5.0_linux-${arch}.run"
  PTO_ISA_PKG="${DOWNLOAD_DIR}/${filename}"

  mkdir -p  "$DOWNLOAD_DIR"

  for date in "${try_dates[@]}"; do
    local url="${base_url}/${date}/ubuntu_${arch}/${filename}"

    if [[ -f "PTO_ISA_PKG" ]]; then
      info "pto isa package exist: $PTO_ISA_PKG"
      return 0
    fi

    info "downloading $url"

    if wget -O "$PTO_ISA_PKG" "$url" --no-check-certificate -q; then
      info "download successful!"
      return 0
    else
      warn "download failed!: $url"
    fi
  done

  error "please check network!"
  exit 1
}

install_pto_isa_run() {
  local toolkit_path="$1"
  local install_path="${toolkit_path}"

  chmod +x "$PTO_ISA_PKG"
  "$PTO_ISA_PKG" --full --quiet --install-path="$install_path"

  if [[ $? -ne 0 ]]; then
    error "install PTO ISA package failed!"
    exit 1
  fi
  info "install PTO ISA package success!"
}

install_gtest() {
  git config --global http.sslverify false
  git clone https://github.com/google/googletest.git -b v1.14.x
  cd googletest
  mkdir build
  cd build
  cmake .. -DCMAKE_CXX_FLAGS="-fPIC"

  if make -j$(nproc); then
    info "build gtest success!"
  else
    error "build gtest failed!"
    exit 1
  fi

  if sudo make install; then
    info "install gtest success!"
  else
    error "install gtest failed!"
    exit 1
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    error "usage: $0 <toolkit_install_path> [toolkit_package_path]"
    exit 1
  fi

  TOOLKIT_INSTALL_PATH="$1"
  TOOLKIT_PACKAGE_PATH="$2"

  local ARCH=$(detect_arch)
  info "arch:$ARCH"

  if check_toolkit_installed "$TOOLKIT_INSTALL_PATH"; then
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$DOWNLOAD_DIR" "$TOOLKIT_INSTALL_PATH"
  else
    if [[ -z "$TOOLKIT_PACKAGE_PATH" ]]; then
      error "no toolkit package info"
      exit 1
    fi

    install_toolkit "$TOOLKIT_PACKAGE_PATH" "$TOOLKIT_INSTALL_PATH"
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$TOOLKIT_INSTALL_PATH"
  fi

  install_gtest

  info "set environment successfully!"
}

main "$@"
