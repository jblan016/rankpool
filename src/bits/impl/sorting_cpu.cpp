// @file sorting_cpu.cpp
// @brief Pooling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

//mod. by j.b. <2017>
/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "sorting.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>

/* ---------------------------------------------------------------- */
/*                                               Max sorting helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{
  inline acc_max(int sortHeight, int sortWidth, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                           Average sorting helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_sum
{
  inline acc_sum(int sortHeight, int sortWidth, type derOutput = 0)
  :
  value(0),
  scale(type(1)/type(sortHeight*sortWidth)),
  derOutput(derOutput)
  { }

  inline void accumulate_forward(type x) {
    value += x ;
  }

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale ;
} ;

/* ---------------------------------------------------------------- */
/*                                                sorting_*_forward */
/* ---------------------------------------------------------------- */

/*
 Reverse accumulation style (better for writing).
 - pick an input coordiante xi; goal is to compute dz/dxi
 - look for all the sorts Pj that cointain xi
 -  compute dfj/dxi (Pj)
 -  accumulate into dz/dxi += dz/dfj dfj/dxi (Pj)

 The advantage of this method is that dz/dxi can be processed in parallel
 without conflicts from other threads writing on different dz/dxi. The
 disadvantage is that for eac xi we need to know dfj/dxi (Pj) for all
 the sorts Pj that contain xi. Barring special cases (e.g. linear) this
 usually requires additional information to be available. For instance,
 for max sorting knowing the output in addition to the input of the
 sorting operator.

 Direct accumulation style.
 - pick an output coordiante fj and its sort Pj
 - for all the input point xi in the sort Pj
 - compute dfj/dxi (Pj)
 - accumulate to dz/dxi += dz/dfj dfj/dxi (Pj)

 The difference with the last method is that different output sorts Pj
 will share several input pixels xi; hence this will cause write conflicts if
 Pj are processed in parallel.
 */

template<typename type, typename Accumulator> static inline void
sorting_forward_cpu(type* sorted,
                    type const* data,
                    size_t width, size_t height, size_t depth,
                    size_t windowWidth, size_t windowHeight,
                    size_t strideX, size_t strideY,
                    size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{
  int sortedWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int sortedHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < sortedHeight; ++y) {
      for (int x = 0; x < sortedWidth; ++x) {
        int x1 = x * (signed)strideX - (signed)padLeft ;
        int y1 = y * (signed)strideY - (signed)padTop ;
        int x2 = std::min(x1 + windowWidth, width) ;
        int y2 = std::min(y1 + windowHeight, height) ;
        x1 = std::max(x1, 0) ;
        y1 = std::max(y1, 0) ;
        Accumulator acc(y2 - y1, x2 - x1) ;
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            acc.accumulate_forward(data[v * width + u]) ;
          }
        }
        sorted[y * sortedWidth + x] = acc.done_forward() ;
      }
    }
    data += width*height ;
    sorted += sortedWidth*sortedHeight ;
  }
}

/* ---------------------------------------------------------------- */
/*                                               sorting_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

/* Todo: transpose */

template<typename type, typename Accumulator> static inline void
sorting_backward_cpu(type* derData,
                     type const* data,
                     type const* derPooled,
                     size_t width, size_t height, size_t depth,
                     size_t windowWidth, size_t windowHeight,
                     size_t strideX, size_t strideY,
                     size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{
  int sortedWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int sortedHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < sortedHeight; ++y) {
      for (int x = 0; x < sortedWidth; ++x) {
        int x1 = x * (signed)strideX - (signed)padLeft ;
        int y1 = y * (signed)strideY - (signed)padTop ;
        int x2 = std::min(x1 + windowWidth, width) ;
        int y2 = std::min(y1 + windowHeight, height) ;
        x1 = std::max(x1, 0) ;
        y1 = std::max(y1, 0) ;
        Accumulator acc(y2 - y1, x2 - x1, derPooled[y * sortedWidth + x]) ;
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            acc.accumulate_backward(&data[v * width + u],
                                    &derData[v * width + u]) ;
          }
        }
        acc.done_backward() ;
      }
    }
    data += width*height ;
    derData += width*height ;
    derPooled += sortedWidth*sortedHeight ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct sorting_max<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* sorted,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t sortHeight, size_t sortWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      sorting_forward_cpu<type, acc_max<type> > (sorted,
                                                 data,
                                                 height, width, depth,
                                                 sortHeight, sortWidth,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t sortHeight, size_t sortWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      sorting_backward_cpu<type, acc_max<type> > (derData,
                                                  data, derOutput,
                                                  height, width, depth,
                                                  sortHeight, sortWidth,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // sorting_max

  template <typename type>
  struct sorting_average<vl::VLDT_CPU, type>
  {

    static vl::ErrorCode
    forward(type* sorted,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t sortHeight, size_t sortWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      sorting_forward_cpu<type, acc_sum<type> > (sorted,
                                                 data,
                                                 height, width, depth,
                                                 sortHeight, sortWidth,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* derPooled,
             size_t height, size_t width, size_t depth,
             size_t sortHeight, size_t sortWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      sorting_backward_cpu<type, acc_sum<type> > (derData,
                                                  NULL, derPooled,
                                                  height, width, depth,
                                                  sortHeight, sortWidth,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // sorting_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::sorting_max<vl::VLDT_CPU, float> ;
template struct vl::impl::sorting_average<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::sorting_max<vl::VLDT_CPU, double> ;
template struct vl::impl::sorting_average<vl::VLDT_CPU, double> ;
#endif

