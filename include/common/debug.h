/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
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
  #include "cpu_sim/tile_offsets.hpp"

  template<typename GT>
  void printRawGT(GT& tensor, const std::string name = "", int elementWidth=5, int maxR=INT32_MAX, int maxC=INT32_MAX) {
      auto rows = tensor.GetShape(3);
      auto cols = tensor.GetShape(4);
      auto stride3 = std::max(tensor.GetStride(3), tensor.GetStride(4));
      std::cout << std::format("{}: {} x {} x {} x {} x {}", name, tensor.GetShape(0), tensor.GetShape(1), tensor.GetShape(2), tensor.GetShape(3), tensor.GetShape(4)) << std::endl;
      for(int i=0; i<tensor.GetShape(0); i++) {
          for(int j=0; j<tensor.GetShape(1); j++) {
              for(int k=0; k<tensor.GetShape(2); k++) {
                  std::cout << std::format("    {}, {}, {}, r, c:\n",i,j,k);

                  for(int y=0; y<rows && y<maxR; y++) {
                      for(int x=0; x<cols && x<maxC; x++) {
                          auto val = tensor.data()[i*tensor.GetStride(0) + j*tensor.GetStride(1)+k*tensor.GetStride(2)+y*stride3+x];
                          if constexpr(std::is_integral_v<typename GT::DType>) {
                              std::cout << std::format("{:{}} ", val,elementWidth);
                          } else {
                              std::cout << std::format("{:{}.2f} ", (val < 1e-20 ? 0 : val), elementWidth);
                          }
                      }
                      if(maxC < cols) {
                        std::cout << " ...";
                      }
                      std::cout << std::endl;
                  }
                  if(maxR < rows) {
                    std::cout << "..." << std::endl;
                  }
              }
          }
      }
  }

    template<typename TL>
    void printRawTile(TL& tile, const std::string name = "", int elementWidth=5, int maxR=INT32_MAX, int maxC=INT32_MAX) {
        std::cout << std::format("{}: {} x {} (Full: {} x {}) (RxC)", name, tile.GetValidRow(), tile.GetValidCol(), tile.Rows, tile.Cols) << std::endl;
        for(int y=0; y<tile.GetValidRow() && y<maxR; y++){
            for(int x=0; x<tile.GetValidCol() && x<maxC; x++) {
                if constexpr(std::is_integral_v<typename TL::DType>) {
                    std::cout << std::format("{:{}} ", tile.data()[y*tile.Cols+x],elementWidth);
                } else {
                    std::cout << std::format("{:{}.2f} ",tile.data()[y*tile.Cols+x] <1e-20 ? 0 : tile.data()[y*tile.Cols+x],elementWidth);
                }
            }
            if(maxC < tile.GetValidCol()) {
              std::cout << " ...";
            }
            std::cout << std::endl;
        }
        if(maxR < tile.GetValidRow()) {
          std::cout << "..." << std::endl;
        }
    }
    
    template<typename TL>
    void printTile(TL& tile, const std::string name = "", int elementWidth=5, int maxR=INT32_MAX, int maxC=INT32_MAX) {
        std::cout << std::format("{}: {} x {} (Full: {} x {}) (RxC)", name, tile.GetValidRow(), tile.GetValidCol(), tile.Rows, tile.Cols) << std::endl;
        for(int y=0; y<tile.GetValidRow() && y<maxR; y++){
            for(int x=0; x<tile.GetValidCol() && x<maxC; x++) {
                auto offset = pto::GetTileElementOffset<TL>(y,x);
                if constexpr(std::is_integral_v<typename TL::DType>) {
                    std::cout << std::format("{:{}} ", tile.data()[offset],elementWidth);
                } else {
                    std::cout << std::format("{:{}.2f} ",tile.data()[offset] <1e-20 ? 0 : tile.data()[offset],elementWidth);
                }
            }
            if(maxC < tile.GetValidCol()) {
              std::cout << " ...";
            }
            std::cout << std::endl;
        }
        if(maxR < tile.GetValidRow()) {
          std::cout << "..." << std::endl;
        }
    }    

    template<typename T>
    void printRawMemory(T * buf, size_t sz, const std::string name = "", int elementWidth=10, int elementsPerRow=8) {
        std::cout << std::format("{}: {}", name, (size_t)buf) << std::endl;
        for(int i=0; i<sz; i++){
            if(i % elementsPerRow == 0) {
              std::cout << std::endl << std::format("{:6x}: ", i);
            }
            if constexpr(std::is_integral_v<T>) {
                std::cout << std::format("{:{}} ", buf[i], elementWidth);
            } else {
                std::cout << std::format("{:{}.2f} ", buf[i] <1e-20 ? 0 : buf[i], elementWidth);
            }
        }
        std::cout << std::endl;
    }

#endif

#endif