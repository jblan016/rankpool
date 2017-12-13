// @file nnsorting_blas.hpp
// @brief Sorting block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnsorting_cudnn__
#define __vl__nnsorting_cudnn__

#include "../nnsorting.hpp"
#include "../data.hpp"
#include "cudnn.h"


namespace vl { namespace impl {

  // todo: data type should be handled internally?

  template<vl::DataType dataType>
  struct nnsorting_cudnn
  {
    static vl::ErrorCode
    forward(Context& context,
            Tensor output,
            Tensor data,
            vl::SortingMethod method,
            int sortHeight, int sortWidth,
            int strideY, int strideX,
            int padTop, int padBottom,
            int padLeft, int padRight) ;

    static vl::ErrorCode
    backward(Context& context,
             Tensor derData,
             Tensor data,
             Tensor output,
             Tensor derOutput,
             vl::SortingMethod method,
             int sortHeight, int sortWidth,
             int strideY, int strideX,
             int padTop, int padBottom,
             int padLeft, int padRight) ;
  };

} }

#endif /* defined(__vl__nnsorting_cudnn__) */
