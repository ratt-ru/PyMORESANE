import numpy as np
import multiprocessing as mp
import ctypes

try:
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
except:
    print "Pycuda unavailable - GPU mode will fail."

def iuwt_decomposition(in1, scale_count, scale_adjust=0, mode='ser', core_count=2, store_smoothed=False):
    """
    This function serves as a handler for the different implementations of the IUWT decomposition. I allows the
    different methods to be used almost interchangeably.

    INPUTS:
    in1                 (no default):       Array on which the decomposition is to be performed.
    scale_count         (no default):       Maximum scale to be considered.
    scale_adjust        (default=0):        Adjustment to scale value if first scales are of no interest.
    mode                (default='ser'):    Implementation of the IUWT to be used.
    core_count          (default=1):        Additional option for multiprocessing - specifies core count.
    store_smoothed      (default=False):    Boolean specifier for whether the smoothed image is stored or not.

    OUTPUTS:
    Returns the decomposition.
    """

    if mode=='ser':
        return ser_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed)
    elif mode=='mp':
        return mp_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, core_count)
    elif mode=='gpu':
        return gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed)

def iuwt_recomposition(in1, scale_adjust=0, mode='ser', core_count=1):
    """
    This function serves as a handler for the different implementations of the IUWT decomposition. It allows the
    different methods to be used almost interchangeably.

    INPUTS:
    in1                 (no default):       Array on which the decomposition is to be performed.
    scale_adjust        (no default):       Number or omitted scales.
    mode                (default='ser')     Implementation of the IUWT to be used.
    core_count          (default=1)         Additional option for multiprocessing - specifies core count.

    OUTPUTS:
    Returns the decomposition.
    """

    if mode=='ser':
        return ser_iuwt_recomposition(in1, scale_adjust)
    elif mode=='mp':
        return mp_iuwt_recomposition(in1, scale_adjust, core_count)
    elif mode=='gpu':
        return gpu_iuwt_recomposition(in1, scale_adjust)

def ser_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for a single CPU.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.

    OUTPUTS:
    detail_coeffs       (no default): Array containing the detail coefficients and the smoothed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter for use in the a trous algorithm.

    # Initialises a zero array to store the coefficients.

    if store_smoothed:
        detail_coeffs = np.empty([scale_count-scale_adjust+1, in1.shape[0], in1.shape[1]])
    else:
        detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    C0 = in1    # Sets the initial value to be the image.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C = ser_a_trous(C0, wavelet_filter, i)
            C0 = C

    # The following loop calculates the wavelet coefficients based on values of a trous algorithm for C0 and C. C0 is
    # resassigned the value of C on each loop.

    for i in range(scale_adjust,scale_count):
        C = ser_a_trous(C0, wavelet_filter, i)             # Approximation coefficients.
        C1 = ser_a_trous(C, wavelet_filter, i)
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                          # Detail coefficients.
        C0 = C

    if store_smoothed:
        detail_coeffs[scale_count-scale_adjust,:,:] = C      # The coeffs at value scale_count are assigned to the last value of C.

    return detail_coeffs

def ser_iuwt_recomposition(in1, scale_adjust):
    """
    This function calls the a trous algorithm code to recompose the input into a single image. This is the serial
    implementation of the isotropic undecimated wavelet transform recomposition.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of truncated array pages.

    OUTPUTS:
    recomposiiton   (no default):   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter for use in the a trous algorithm.

    # Following checks that in1 is not simply a single 2D array. This just ensures that the recomposition is
    # the correct size.

    if len(in1.shape)>2:
        scale_count = in1.shape[0]
        recomposition = np.zeros([in1.shape[1], in1.shape[2]])
    else:
        scale_count = 1
        recomposition = np.zeros([in1.shape[0], in1.shape[1]])

    # Maxscale is adjusted by the value of scaleadjust to give its true value.

    max_scale = scale_count + scale_adjust

    # The following loops are calls to the atrous algorithm with the addition of scaladjust to reduce the number of
    # iterations.

    for i in range(max_scale-1, scale_adjust-1, -1):
        recomposition = ser_a_trous(recomposition, wavelet_filter, i) + in1[i-scale_adjust,:,:]

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            recomposition = ser_a_trous(recomposition, wavelet_filter, i)

    return recomposition

def ser_a_trous(C0, filter, scale):
    """
    The following is an implementation of the a trous algorithm for wavelet decomposition. Accepts the following
    parameters:

    filter      (no default): The filter which is applied to the components of the transform.
    C0          (no default): The current array being for decomposition.
    scale       (no default): The scale for which the decomposition is being carried out.

    """

    C0size = C0.shape

    v = np.empty(C0size)
    C1 = np.empty(C0size)

    v = filter[2]*C0

    v[(2**(scale+1)):,:] += filter[0]*C0[:-(2**(scale+1)),:]
    v[:(2**(scale+1)),:] += filter[0]*C0[(2**(scale+1))-1::-1,:]

    v[(2**scale):,:] += filter[1]*C0[:-(2**scale),:]
    v[:(2**scale),:] += filter[1]*C0[(2**scale)-1::-1,:]

    v[:-(2**scale),:] += filter[3]*C0[(2**scale):,:]
    v[-(2**scale):,:] += filter[3]*C0[:-(2**scale)-1:-1,:]

    v[:-(2**(scale+1)),:] += filter[4]*C0[(2**(scale+1)):,:]
    v[-(2**(scale+1)):,:] += filter[4]*C0[:-(2**(scale+1))-1:-1,:]

    C1 = filter[2]*v

    C1[:,(2**(scale+1)):] += filter[0]*v[:,:-(2**(scale+1))]
    C1[:,:(2**(scale+1))] += filter[0]*v[:,(2**(scale+1))-1::-1]

    C1[:,(2**scale):] += filter[1]*v[:,:-(2**scale)]
    C1[:,:(2**scale)] += filter[1]*v[:,(2**scale)-1::-1]

    C1[:,:-(2**scale)] += filter[3]*v[:,(2**scale):]
    C1[:,-(2**scale):] += filter[3]*v[:,:-(2**scale)-1:-1]

    C1[:,:-(2**(scale+1))] += filter[4]*v[:,(2**(scale+1)):]
    C1[:,-(2**(scale+1)):] += filter[4]*v[:,:-(2**(scale+1))-1:-1]

    return C1

def mp_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, core_count):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for multiple CPUs.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.
    core_count          (no default):   Indicates the number of cores to be used.

    OUTPUTS:
    detail_coeffs       (no default): Array containing the detail coefficients and the smoothed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # A trous algorithm filter - B3 spline.

    C0 = in1                                            # Sets the initial value to be the image.

    # Initialises a zero array to store the coefficients.

    if store_smoothed:
        detail_coeffs = np.empty([scale_count-scale_adjust+1, in1.shape[0], in1.shape[1]])
    else:
        detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C = mp_a_trous(C0, wavelet_filter, i, core_count)
            C0 = C

    # The following loop calculates the wavelet coefficients using the a trous algorithm for C0 and C. C0 is
    # reassigned the value of C on each loop - C0 is always the smoothest version of the input image.

    for i in range(scale_adjust,scale_count):
        C = mp_a_trous(C0, wavelet_filter, i, core_count)              # Approximation coefficients.
        C1 = mp_a_trous(C, wavelet_filter, i, core_count)
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                                      # Detail coefficients.
        C0 = C

    if store_smoothed:
        detail_coeffs[scale_count-scale_adjust,:,:] = C     # The coeffs at value scale_count are assigned to the last
                                                            # value of C which corresponds to the smoothest version of
                                                            # the input image.
    return detail_coeffs

def mp_iuwt_recomposition(in1, scale_adjust, core_count):
    """
    This function calls the a trous algorithm code to recompose the input into a single image. This is the
    multiprocessing implementation of the isotropic undecimated wavelet transform recomposition.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of truncated array pages.
    core_count      (no default):   Indicates the number of cores to be used.

    OUTPUTS:
    recomposiiton   (no default):   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter for use in the a trous algorithm.

    # Following checks that in1 is not simply a single 2D array.

    if len(in1.shape)>2:
        scale_count = in1.shape[0]
        recomposition = np.zeros([in1.shape[1], in1.shape[2]])
    else:
        scale_count = 1
        recomposition = np.zeros([in1.shape[0], in1.shape[1]])

    # Maxscale is adjusted by the value of scaleadjust to give its true value.

    max_scale = scale_count + scale_adjust

    # The following loops are calls to the atrous algorithm with the addition of scaladjust to reduce the number of
    # iterations.

    for i in range(max_scale-1, scale_adjust-1, -1):
        recomposition = mp_a_trous(recomposition, wavelet_filter, i, core_count) + in1[i-scale_adjust,:,:]

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            recomposition = ser_a_trous(recomposition, wavelet_filter, i, core_count)

    return recomposition

def mp_a_trous(C0, wavelet_filter, scale, core_count):
    """
    This is a reimplementation of the a trous algorithm which makes use of multiprocessing. In particular,
    it divides the input array of dimensions NxN into M smaller arrays of dimensions (N/M)xN, where M is the
    number of cores which are to be used.

    INPUTS:
    C0              (no default): The current array which is to be decomposed.
    wavelet_filter  (no default): The filter which is applied during the decomposition.
    scale           (no default): The scale at which decomposition is to be carried out.
    core_count      (no default): The number of CPU cores over which the task should be divided.

    OUTPUTS:
    shared_array    (no default): The decomposed array.
    """

    # Creates an array which may be accessed by multiple processes.

    shared_array_base = mp.Array(ctypes.c_float, C0.shape[0]**2, lock=False)
    shared_array = np.frombuffer(shared_array_base, dtype=ctypes.c_float)
    shared_array = shared_array.reshape(C0.shape)
    shared_array[:,:] = C0

    # Division of the problem and allocation of processes to cores.

    processes = []

    for i in range(core_count):
        process = mp.Process(target = mp_a_trous_kernel, args = (shared_array, wavelet_filter, scale, i,
                                                     C0.shape[0]//core_count, 'row',))
        process.start()
        processes.append(process)

    for i in processes:
        i.join()

    processes = []

    for i in range(core_count):
        process = mp.Process(target = mp_a_trous_kernel, args = (shared_array, wavelet_filter, scale, i,
                                                     C0.shape[1]//core_count, 'col',))
        process.start()
        processes.append(process)

    for i in processes:
        i.join()

    return shared_array

def mp_a_trous_kernel(C0, wavelet_filter, scale, slice_ind, slice_width, r_or_c="row"):
    """
    This is the convolution step of the a trous algorithm.

    INPUTS:
    C0              (no default):       The current array which is to be decomposed.
    wavelet_filter  (no default):       The filter which is applied during the decomposition.
    scale           (no default):       The scale at which decomposition is to be carried out.
    slice_ind       (no default):       The index of the particular slice in which the decomposition is being performed.
    slice_width     (no default):       The number of elements of the shorter dimension of the array for decomposition.
    r_or_c          (default = "row"):  Indicates whether strips are rows or columns.

    OUTPUTS:
    NONE - C0 is a mutable array which this function alters. No value is returned.

    """

    lower_bound = slice_ind*slice_width
    upper_bound = (slice_ind+1)*slice_width

    if r_or_c == "row":
        row_conv = wavelet_filter[2]*C0[:,lower_bound:upper_bound]

        row_conv[(2**(scale+1)):,:] += wavelet_filter[0]*C0[:-(2**(scale+1)),lower_bound:upper_bound]
        row_conv[:(2**(scale+1)),:] += wavelet_filter[0]*C0[(2**(scale+1))-1::-1,lower_bound:upper_bound]

        row_conv[(2**scale):,:] += wavelet_filter[1]*C0[:-(2**scale),lower_bound:upper_bound]
        row_conv[:(2**scale),:] += wavelet_filter[1]*C0[(2**scale)-1::-1,lower_bound:upper_bound]

        row_conv[:-(2**scale),:] += wavelet_filter[3]*C0[(2**scale):,lower_bound:upper_bound]
        row_conv[-(2**scale):,:] += wavelet_filter[3]*C0[:-(2**scale)-1:-1,lower_bound:upper_bound]

        row_conv[:-(2**(scale+1)),:] += wavelet_filter[4]*C0[(2**(scale+1)):,lower_bound:upper_bound]
        row_conv[-(2**(scale+1)):,:] += wavelet_filter[4]*C0[:-(2**(scale+1))-1:-1,lower_bound:upper_bound]

        C0[:,lower_bound:upper_bound] = row_conv

    elif r_or_c == "col":
        col_conv = wavelet_filter[2]*C0[lower_bound:upper_bound,:]

        col_conv[:,(2**(scale+1)):] += wavelet_filter[0]*C0[lower_bound:upper_bound,:-(2**(scale+1))]
        col_conv[:,:(2**(scale+1))] += wavelet_filter[0]*C0[lower_bound:upper_bound,(2**(scale+1))-1::-1]

        col_conv[:,(2**scale):] += wavelet_filter[1]*C0[lower_bound:upper_bound,:-(2**scale)]
        col_conv[:,:(2**scale)] += wavelet_filter[1]*C0[lower_bound:upper_bound,(2**scale)-1::-1]

        col_conv[:,:-(2**scale)] += wavelet_filter[3]*C0[lower_bound:upper_bound,(2**scale):]
        col_conv[:,-(2**scale):] += wavelet_filter[3]*C0[lower_bound:upper_bound,:-(2**scale)-1:-1]

        col_conv[:,:-(2**(scale+1))] += wavelet_filter[4]*C0[lower_bound:upper_bound,(2**(scale+1)):]
        col_conv[:,-(2**(scale+1)):] += wavelet_filter[4]*C0[lower_bound:upper_bound,:-(2**(scale+1))-1:-1]

        C0[lower_bound:upper_bound,:] = col_conv

def gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for GPUs.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.

    OUTPUTS:
    detail_coeffs       (no default): Array containing the detail coefficients and the smoothed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    # Initialises a zero array to store the coefficients.

    if store_smoothed:
        detail_coeffs = np.empty([scale_count-scale_adjust+1, in1.shape[0], in1.shape[1]])
    else:
        detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    C0 = gpuarray.to_gpu_async(in1.astype(np.float32))                  # Sends the initial value to the GPU.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C = gpu_a_trous(C0, wavelet_filter, i)
            C0 = C

    # The following loop calculates the detail coefficients using the a trous algorithm for C0 and C. C0 is
    # reassigned the value of C on each loop - C is the smoothest version of the input image.

    for i in range(scale_adjust,scale_count):
        C = gpu_a_trous(C0, wavelet_filter, i)                      # Wavelet coefficients.
        C1 = gpu_a_trous(C, wavelet_filter, i)
        detail_coeffs[i-scale_adjust,:,:] = (C0 - C1).get_async()   # Calculates and stores the detail coefficients.
        C0 = C

    if store_smoothed:
        detail_coeffs[scale_count-scale_adjust,:,:] = C.get()   # The coeffs at value scale_count are assigned to
                                                                # the last value of C which corresponds to the
                                                                # smoothest version of the input image.
    return detail_coeffs

def gpu_iuwt_recomposition(in1, scale_adjust):
    """
    This function calls the a trous algorithm code to recompose the input into a single image. This is the
    multiprocessing implementation of the isotropic undecimated wavelet transform recomposition.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of truncated array pages.

    OUTPUTS:
    recomposiiton   (no default):   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))

    # Following checks that in1 is not simply a single 2D array.

    if len(in1.shape)>2:
        scale_count = in1.shape[0]
        recomposition = gpuarray.zeros([in1.shape[1], in1.shape[2]], np.float32)
    else:
        scale_count = 1
        recomposition = gpuarray.zeros([in1.shape[0], in1.shape[1]], np.float32)

    # Maxscale is adjusted by the value of scaleadjust to give its true value.

    max_scale = scale_count + scale_adjust

    # The following loops are calls to the atrous algorithm with the addition of scaladjust to reduce the number of
    # iterations.

    for i in range(max_scale-1, scale_adjust-1, -1):
        recomposition = gpu_a_trous(recomposition, wavelet_filter, i)[:,:] + gpu_in1[i-scale_adjust,:,:]

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            recomposition = gpu_a_trous(recomposition, wavelet_filter, i)

    return recomposition.get()

def gpu_a_trous(in1, wavelet_filter, scale):
    """
    This function makes use of the GPU and PyCUDA to implement a fast version of the a trous algorithm. May be
    slow on initial run when kernel compiles.

    INPUTS:
    in1                 (no default): GPUArray on which the decomposition is to be performed.
    scale               (no default): Scale of the decomposition.
    wavelet_filter      (no default): The filter which is applied during the decomposition.

    OUTPUTS:
    in1_copy            (no default): GPUArray containing the decomposition.
    """

    ker1 = SourceModule("""
                        __global__ void gpu_a_trous_row_kernel(float *out1, float *in1, int *scale, float *wfil)
                        {
                          const int len = gridDim.y*blockDim.y;
                          const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                          const int j = (blockDim.y * blockIdx.y + threadIdx.y)*len;
                          const int tid = i + j;
                          int row = tid / len;

                          out1[tid] = wfil[2]*in1[tid];

                          if (tid < scale[0]*len)
                            {
                                out1[tid] += wfil[0]*in1[i + len*(scale[0] - row - 1)];
                            }
                          else
                            {
                                out1[tid] += wfil[0]*in1[tid - scale[0]*len];
                            }

                          if (tid < scale[1]*len)
                            {
                                out1[tid] += wfil[1]*in1[i + len*(scale[1] - row - 1)];
                            }
                          else
                            {
                                out1[tid] += wfil[1]*in1[tid - scale[1]*len];
                            }

                          if (tid >= len*len - scale[1]*len)
                            {
                                out1[tid] += wfil[3]*in1[len*(len - (row + scale[1])%(len-1)) + i];
                            }
                          else
                            {
                                out1[tid] += wfil[3]*in1[tid + scale[1]*len];
                            }

                          if (tid >= len*len - scale[0]*len)
                            {
                                out1[tid] += wfil[4]*in1[len*(len - (row + scale[0])%(len-1)) + i];
                            }
                          else
                            {
                                out1[tid] += wfil[4]*in1[tid + scale[0]*len];
                            }

                        }
                       """)

    ker2 = SourceModule("""
                        __global__ void gpu_a_trous_col_kernel(float *out1, float *in1, int *scale, float *wfil)
                        {
                          const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                          const int j = (blockDim.y * blockIdx.y + threadIdx.y)*gridDim.y*blockDim.y;
                          const int tid = i + j;
                          const int len = gridDim.x*blockDim.x;
                          int col = 0;

                          out1[tid] = wfil[2]*in1[tid];

                          if (i < scale[0])
                            {
                                out1[tid] += wfil[0]*in1[j - (i - (scale[0] - 1))];
                            }
                          else
                            {
                                out1[tid] += wfil[0]*in1[tid - scale[0]];
                            }

                          if (i < scale[1])
                            {
                                out1[tid] += wfil[1]*in1[j - (i - (scale[1] - 1))];
                            }
                          else
                            {
                                out1[tid] += wfil[1]*in1[tid - scale[1]];
                            }

                          if (j==0)
                            {
                                col = tid - len;
                            }
                          else
                            {
                                col = (tid % j) - len;
                            }

                          if (col >= -scale[1])
                            {
                                out1[tid] += wfil[3]*in1[j + len - (col + scale[1] + 1)];
                            }
                          else
                            {
                                out1[tid] += wfil[3]*in1[tid + scale[1]];
                            }

                          if (col >= -scale[0])
                            {
                                out1[tid] += wfil[4]*in1[j + len - (col + scale[0] + 1)];
                            }
                          else
                            {
                                out1[tid] += wfil[4]*in1[tid + scale[0]];
                            }

                        }
                       """)

    scale = np.array([2**(scale+1), 2**scale], dtype=np.int32)
    scale = gpuarray.to_gpu_async(scale)

    in1_copy = in1.copy()                   # A copy is used as to preserve the existing values.
    out1 = gpuarray.empty_like(in1_copy)

    gpu_a_trous_row_kernel = ker1.get_function("gpu_a_trous_row_kernel")

    gpu_a_trous_row_kernel(out1, in1_copy, scale, wavelet_filter,
                       block=(32,32,1), grid=(in1.shape[0]//32, in1.shape[1]//32))

    gpu_a_trous_col_kernel = ker2.get_function("gpu_a_trous_col_kernel")

    gpu_a_trous_col_kernel(in1_copy, out1, scale, wavelet_filter,
                       block=(32,32,1), grid=(in1.shape[0]//32, in1.shape[1]//32))

    return in1_copy

# if __name__ == "__main__":
#     test = np.random.randn(512,512).astype(np.float32)
#     result1 = iuwt_decomposition(test, 4, scale_adjust=2, mode="mp", store_smoothed=True)
#
#     result2 = iuwt_decomposition(test, 4, scale_adjust=2, mode="mp")
#
#     print np.max(result2-result1[:2,:,:])
#     print result1.shape
#     print result2.shape