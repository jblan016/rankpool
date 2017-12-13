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
#include <cub/cub.cuh>

/* ---------------------------------------------------------------- */
/*                                              sorting_max_forward */
/* ---------------------------------------------------------------- */

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD> __global__ void
sorting_kernel
(T* sorted,
 const T* data,
 const int Width,
 const int Height,
 const int windowWidth,
 const int windowHeight,
 const int strideWidth,
 const int strideHeight,
 const int BoxesInWidth
 )
{
	const int numElemsPerArray = BLOCK_THREADS * ITEMS_PER_THREAD;
    // --- Shared memory allocation
	__shared__ T sharedMemoryValueArray[numElemsPerArray];
    // --- Specialize BlockStore and BlockRadixSort collective types
    typedef cub::BlockRadixSort <int , BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT; //led dernier argument est pour le rang

    // --- Allocate type-safe, repurposable shared memory for collectives
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
// section d_in->d_out
    int offsetk= blockIdx.y*gridDim.x*numElemsPerArray;
    int offsetj=blockIdx.x/BoxesInWidth;
    offsetj=offsetj*strideWidth;
    int offseti=blockIdx.x%BoxesInWidth;
        offseti=offseti*strideHeight;
//
    int block_offset = numElemsPerArray * blockIdx.x+offsetk;
    int arrayAddress = 0;
    int windowoffsetj = 0;
    // --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++){
    	arrayAddress = threadIdx.x * ITEMS_PER_THREAD + k;
    	windowoffsetj=(arrayAddress/windowHeight + offsetj) * Height;
    	sharedMemoryValueArray[arrayAddress]  = data[arrayAddress%windowHeight+windowoffsetj+offseti+offsetk];//loads array
    }
    __syncthreads();

    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage).Sort(*static_cast<T(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryValueArray + (threadIdx.x * ITEMS_PER_THREAD))));

    __syncthreads();

    // --- Write data from shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++){
    	arrayAddress = threadIdx.x * ITEMS_PER_THREAD + k;
    	windowoffsetj=(arrayAddress/windowHeight + offsetj) * Height;
    	sorted[block_offset + arrayAddress] = sharedMemoryValueArray[arrayAddress];
    }

}



/* ---------------------------------------------------------------- */
/*                                             sorting_max_backward */
/* ---------------------------------------------------------------- */



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
     This is bad(why the following?), but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derSorted[sortedIndex]) ;
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

 

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::sorting_max<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::sorting_max<vl::VLDT_GPU, double> ;
#endif

