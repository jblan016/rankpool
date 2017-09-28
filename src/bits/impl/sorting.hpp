// @file sorting.hpp
// @brief Sorting block implementation
// @author Andrea Vedaldi
// @author Karel Lenc

//mod j.b.<2017>
/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_SORTING_H
#define VL_SORTING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct pooling_max {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::ErrorCode
    backward(data_type* derData,
             data_type const* data,
             data_type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

  template<vl::DeviceType dev, typename type>
  struct pooling_average {
    typedef type data_type ;

    static vl::ErrorCode
    forward(data_type* output,
            data_type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;

    static vl::ErrorCode
    backward(type* derData,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight) ;
  } ;

} }

#endif /* defined(VL_POOLING_H) */
