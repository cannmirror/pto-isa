#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <stdint.h>

#include <iostream>
#include <type_traits>

#include "common/math_utils.hpp"

namespace pto {
enum class Location {
  Vec,
  Mat,
  Left,
  Right,
  Acc,
  Bias,
  Scaling,
};

enum class BLayout {
  RowMajor = 0,
  ColMajor = 1,
};

enum class SLayout {
  NoneBox = 0,
  RowMajor = 1,
  ColMajor = 2,
};

// returns the memory qualifier for a given location and data type.
// compilation errors occur if the location does not have a specialized version.
template <Location L, typename DType> struct MemoryQualifier;

template <typename DType> struct MemoryQualifier<Location::Vec, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__ubuf__"))) DType;
#else
  using type = __ubuf__ DType *;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Mat, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__cbuf__"))) DType;
#else
  using type = __cbuf__ DType *;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Left, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__ca__"))) DType;
#else
  using type = __ca__ DType *;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Right, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__cb__"))) DType;
#else
  using type = __cb__ DType *;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Acc, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__cc__"))) DType;
#else
  using type = __cc__ DType *;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Bias, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__bt__"))) DType;
#else
  using type = uint64_t;
#endif
};

template <typename DType> struct MemoryQualifier<Location::Scaling, DType> {
#ifdef __PTO_AUTO__
  using type = __attribute((annotate("__fbuf__"))) DType;
#else
  using type = __fbuf__ DType *;
#endif
};

} // namespace pto

#endif