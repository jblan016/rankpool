// @file nnsorting.cu
// @brief Sorting block
// @author Andrea Vedaldi
// @author Jonathan Blanchette 2017

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnsorting.hpp"
#include "impl/sorting.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
#include "impl/nnsorting_cudnn.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnsorting_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(), \
sortHeight, sortWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlSortingAverage : DISPATCH2(deviceType, sorting_average) ; break ; \
case vlSortingMax : DISPATCH2(deviceType, sorting_max) ; break ; \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnsorting_cudnn<dataType>::forward \
(context, output, data, \
method, \
sortHeight, sortWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnsorting_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      SortingMethod method,
                      int sortHeight, int sortWidth,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::VLE_Success) { return status ; }
        if (status != vl::VLE_Unsupported) { return status ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnsorting_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnsorting_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_sorting_average(deviceType, type) \
status = vl::impl::sorting_average<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), \
sortHeight, sortWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH_sorting_max(deviceType, type) \
status = vl::impl::sorting_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), \
sortHeight, sortWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnsorting_backward(Context& context,
                       Tensor derData,
                       Tensor data,
                       Tensor derOutput,
                       SortingMethod method,
                       int sortHeight, int sortWidth,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         Unfortunately CuDNN requires both the input and the output sorting arrays
         to be available for computing derivatives, whereas MatConvNet only requires the input one.
         */
      }
#endif
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("sorting_*::backward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnsorting_backward") ;
}
