/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file debug.h
 * \brief
 */

#ifndef PTO_DEBUG_H
#define PTO_DEBUG_H

#define DEBUG_CHECK(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      cce::printf("[DEBUG CHECK FAILED] %s (File: %s, Line: %d)\n", message,   \
                  __FILE__, __LINE__);                                         \
      return;                                                                  \
    }                                                                          \
  } while (0)

#ifdef _DEBUG
#define PTO_ASSERT(condition, message) DEBUG_CHECK(condition, message)
#else
#define PTO_ASSERT(condition, message) ((void)0)
#endif

#ifdef __CPU_SIM
  template<typename GT>
  void printRawGT(GT& tensor, const std::string name = "", int elementWidth=5) {
      auto rows = tensor.GetShape(3);
      auto cols = tensor.GetShape(4);
      std::cout << std::format("{}: {} x {} x {} x {} x {}", name, tensor.GetShape(0), tensor.GetShape(1), tensor.GetShape(2), tensor.GetShape(3), tensor.GetShape(4)) << std::endl;
      for(int y=0; y<rows; y++){
          for(int x=0; x<cols; x++)
              if constexpr(std::is_integral_v<typename GT::DType>) {
                  std::cout << std::format("{:{}} ", tensor.data()[y*cols+x],elementWidth);
              } else {
                  std::cout << std::format("{:{}.2f} ",tensor.data()[y*cols+x] <1e-20 ? 0 : tensor.data()[y*cols+x],elementWidth);
              }
          std::cout << std::endl;
      }
  }

    template<typename TL>
    void printRawTile(TL& tile, const std::string name = "", int elementWidth=5) {
        std::cout << std::format("{}: {} x {} (Full: {} x {}) (RxC)", name, tile.ValidRow, tile.ValidCol, tile.Rows, tile.Cols) << std::endl;
        for(int y=0; y<tile.ValidRow; y++){
            for(int x=0; x<tile.ValidCol; x++)
                if constexpr(std::is_integral_v<typename TL::DType>) {
                    std::cout << std::format("{:{}} ", tile.data()[y*tile.Cols+x],elementWidth);
                } else {
                    std::cout << std::format("{:{}.2f} ",tile.data()[y*tile.Cols+x] <1e-20 ? 0 : tile.data()[y*tile.Cols+x],elementWidth);
                }
            std::cout << std::endl;
        }
    }

#endif

#endif