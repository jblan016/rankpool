// @file sorting_gpu.cu
// @brief Sorting block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

//mod. by j.b.<2017>
/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "sorting.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
#include <thrust/sort.h>

/* ---------------------------------------------------------------- */
/*                                              sorting_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
sorting_max_kernel
(T* sorted,
 const T* data,
 const int sortedWidth,
 const int sortedHeight,
 const int sortedVolume,
 const int width,
 const int height,
 const int sortWidth,
 const int sortHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int sortedIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (sortedIndex < sortedVolume) {
    int px = sortedIndex ;
    int py = px / sortedWidth ;
    int pz = py / sortedHeight ;
    px %= sortedWidth ;
    py %= sortedHeight ;
    data += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + sortWidth, width) ;
    int y2 = min(y1 + sortHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    T bestValue = data[y1 * width + x1] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        bestValue = max(bestValue, data[y * width + x]) ;
      }
    }
    sorted[sortedIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                          sorting_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
sorting_average_kernel
(T* sorted,
 const T* data,
 const int sortedWidth,
 const int sortedHeight,
 const int sortedVolume,
 const int width,
 const int height,
 const int sortWidth,
 const int sortHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  /* sortedIndex = x + y * sortedWidth + z * (sortedWidth * sortedHeight) */
  int sortedIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (sortedIndex < sortedVolume) {
    int px = sortedIndex ;
    int py = px / sortedWidth ;
    int pz = py / sortedHeight ;
    px %= sortedWidth ;
    py %= sortedHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + sortWidth, width) ;
    int y2 = min(y1 + sortHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T accum = 0;
    T sortSize = (y2 - y1)*(x2 - x1);
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        accum += data[y * width + x] ;
      }
    }
    sorted[sortedIndex] = accum / sortSize ;
  }
}

/* ---------------------------------------------------------------- */
/*                                             sorting_max_backward */
/* ---------------------------------------------------------------- */

#ifdef VLNN_CAFFELIKE_BPSORT
// In order to be able to use this, BP would need to have access to both
// bottom data and sorted data (currently only passed bottom data...)
template <typename T> __global__ void
sorting_max_backward_with_sorted_data
(T* derData,
 const T* data,
 const T* sorted,
 const T* derSorted,
 const int nthreads,
 const int sortedWidth,
 const int sortedHeight,
 const int width,
 const int height,
 const int depth,
 const int sortWidth,
 const int sortHeight,
 const int strideX,
 const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % depth;
    int py1 = (y < sortHeight) ? 0 : (y - sortHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, sortedHeight);
    int px1 = (x < sortWidth) ? 0 : (x - sortWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, sortedWidth);
    T gradient = 0;
    T datum = data[(z * height + y) * width + x];
    sorted += z * sortedHeight * sortedWidth;
    dzdy += z * sortedHeight * sortedWidth;
    for (int py = py1; py < py2; ++py) {
      for (int px = px1; px < px2; ++px) {
        gradient += dzdy[py * sortedWidth + px] *
        (datum == sorted[py * sortedWidth + px]);
      }
    }
    dzdx[index] = gradient;
  }
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// an implementation of atomicAdd() for double (really slow) for older CC
static __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template<typename T> __global__ void
sorting_max_backward_kernel
(T* derData,
 const T* data,
 const T* derSorted,
 const int sortedWidth,
 const int sortedHeight,
 const int sortedVolume,
 const int width,
 const int height,
 const int sortWidth,
 const int sortHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int sortedIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (sortedIndex < sortedVolume) {
    int px = sortedIndex ;
    int py = px / sortedWidth ;
    int pz = py / sortedHeight ;
    px %= sortedWidth ;
    py %= sortedHeight ;
    data += pz * (width*height) ;
    derData += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + sortWidth, width) ;
    int y2 = min(y1 + sortHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        int index = y * width + x ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derSorted[sortedIndex]) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                         sorting_average_backward */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
sorting_average_backward_kernel(T* derData,
                                const T* derSorted,
                                const int nthreads,
                                const int sortedWidth,
                                const int sortedHeight,
                                const int width,
                                const int height,
                                const int depth,
                                const int sortWidth,
                                const int sortHeight,
                                const int strideX,
                                const int strideY,
                                const int padLeft,
                                const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    /* To understand the logic of this piece of code see the
     comments to of the row2im backward kernel */
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - sortWidth ;
    int dy = y_data + padTop - sortHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, sortedWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, sortedHeight - 1) ;
    T accumulator = 0 ;
    derSorted += z * sortedHeight * sortedWidth;
    for (int py = py1 ; py <= py2 ; ++py) {
      for (int px = px1 ; px <= px2 ; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;
        int x2 = min(x1 + sortWidth, width) ;
        int y2 = min(y1 + sortHeight, height) ;
        x1 = max(x1, 0) ;
        y1 = max(y1, 0) ;
        T sortSize = (y2 - y1) * (x2 - x1);
        accumulator += derSorted[py * sortedWidth + px] / sortSize ;
      }
    }
    derData[index] = accumulator ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct sorting_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* sorted,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t sortHeight, size_t sortWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom,
            size_t padLeft, size_t padRight)
    {
      int sortedWidth = (width + (padLeft+padRight) - sortWidth)/strideX + 1 ;
      int sortedHeight = (height + (padTop+padBottom) - sortHeight)/strideY + 1 ;
      int sortedVolume = sortedWidth * sortedHeight * depth ;

      sorting_max_kernel<type>
      <<< divideAndRoundUp(sortedVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (sorted, data,
       sortedHeight, sortedWidth, sortedVolume,
       height, width,
       sortHeight, sortWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
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
      int sortedWidth = (width + (padLeft+padRight) - sortWidth)/strideX + 1 ;
      int sortedHeight = (height + (padTop+padBottom) - sortHeight)/strideY + 1 ;
      int sortedVolume = sortedWidth * sortedHeight * depth ;

      sorting_max_backward_kernel<type>
      <<< divideAndRoundUp(sortedVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, derOutput,
       sortedHeight, sortedWidth, sortedVolume,
       height, width,
       sortHeight, sortWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // sorting_max

  template <typename type>
  struct sorting_average<vl::VLDT_GPU, type>
  {

    static vl::ErrorCode
    forward(type* sorted,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t sortHeight, size_t sortWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int sortedWidth = (width + (padLeft+padRight) - sortWidth)/strideX + 1 ;
      int sortedHeight = (height + (padTop+padBottom) - sortHeight)/strideY + 1 ;
      int sortedVolume = sortedWidth * sortedHeight * depth ;

      sorting_average_kernel<type>
      <<< divideAndRoundUp(sortedVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (sorted, data,
       sortedHeight, sortedWidth, sortedVolume,
       height, width,
       sortHeight, sortWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* derSorted,
             size_t height, size_t width, size_t depth,
             size_t sortHeight, size_t sortWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      int sortedWidth = (width + (padLeft+padRight) - sortWidth)/strideX + 1 ;
      int sortedHeight = (height + (padTop+padBottom) - sortHeight)/strideY + 1 ;
      int dataVolume = width * height * depth ;

      sorting_average_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, derSorted,
       dataVolume,
       sortedHeight, sortedWidth,
       height, width, dataVolume,
       sortHeight, sortWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // sorting_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::sorting_max<vl::VLDT_GPU, float> ;
template struct vl::impl::sorting_average<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::sorting_max<vl::VLDT_GPU, double> ;
template struct vl::impl::sorting_average<vl::VLDT_GPU, double> ;
#endif

