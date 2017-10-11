// @file vl_nnsort.cu
// @brief Sorting block MEX wrapper
// @author Andrea Vedaldi
// @author Karel Lenc
//modded by j.b.<2017>

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnsorting.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_method,
  opt_verbose
} ;

/* options */
VLMXOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"Method",           1,   opt_method            },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int sortWidth ;
  int sortHeight ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  vl::SortingMethod method = vl::vlSortingMax ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("PAD has neither one nor four elements.") ;
        }
        break;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlSortingMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlSortingAverage ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
        }
        break;

      default:
        break ;
    }
  }
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      sortHeight = mxGetPr(in[IN_SIZE])[0] ;
      sortWidth = sortHeight ;
      break ;
    case 2:
      sortHeight = mxGetPr(in[IN_SIZE])[0] ;
      sortWidth = mxGetPr(in[IN_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }

  /* Basic compatibility of Shape */
  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (sortHeight == 0 || sortWidth == 0) {
    mexErrMsgTxt("A dimension of the sorting SIZE is void.") ;
  }
  if (data.getHeight() + (padTop+padBottom) < sortHeight ||
      data.getWidth() + (padLeft+padRight) < sortWidth) {
    mexErrMsgTxt("The sorting window is larger than the DATA (including padding).") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }
  if (padLeft >= sortWidth ||
      padRight >= sortWidth ||
      padTop >= sortHeight  ||
      padBottom >= sortHeight) {
    mexErrMsgTxt("A padding value is larger or equal to the size of the sorting window.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape((data.getHeight() + (padTop+padBottom) - sortHeight)/strideY + 1,
                              (data.getWidth()  + (padLeft+padRight) - sortWidth)/strideX + 1,
                              data.getDepth(),
                              data.getSize()) ;

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and SORT.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnsort: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n") ;
    }
    mexPrintf("vl_nnsort: stride: [%d %d], pad: [%d %d %d %d]\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight) ;
    vl::print("vl_nnsort: data: ", data) ;
    mexPrintf("vl_nnsort: sorting: %d x %d\n", sortHeight, sortWidth);
    mexPrintf("vl_nnsort: method: %s\n", (method == vl::vlSortingMax) ? "max" : "avg") ;
    if (backMode) {
      vl::print("vl_nnsort: derOutput: ", derOutput) ;
      vl::print("vl_nnsort: derData: ", derData) ;
    } else {
      vl::print("vl_nnsort: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnsorting_forward(context,
                                  output, data,
                                  method,
                                  sortHeight, sortWidth,
                                  strideY, strideX,
                                  padTop, padBottom, padLeft, padRight) ;
  } else {
    error = vl::nnsorting_backward(context,
                                   derData, data, derOutput,
                                   method,
                                   sortHeight, sortWidth,
                                   strideY, strideX,
                                   padTop, padBottom, padLeft, padRight) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
