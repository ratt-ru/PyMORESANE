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

def iuwt_decomposition(in1, scale_count, scale_adjust=0, mode='ser', core_count=2, store_smoothed=False,
                       store_on_gpu=False):
    """
    This function serves as a handler for the different implementations of the IUWT decomposition. It allows the
    different methods to be used almost interchangeably.

    INPUTS:
    in1                 (no default):       Array on which the decomposition is to be performed.
    scale_count         (no default):       Maximum scale to be considered.
    scale_adjust        (default=0):        Adjustment to scale value if first scales are of no interest.
    mode                (default='ser'):    Implementation of the IUWT to be used - 'ser', 'mp' or 'gpu'.
    core_count          (default=1):        Additional option for multiprocessing - specifies core count.
    store_smoothed      (default=False):    Boolean specifier for whether the smoothed image is stored or not.
    store_on_gpu        (default=False):    Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    Returns the decomposition with the additional smoothed coefficients if specified.
    """

    if mode=='ser':
        return ser_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed)
    elif mode=='mp':
        return mp_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, core_count)
    elif mode=='gpu':
        return gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, store_on_gpu)

def iuwt_recomposition(in1, scale_adjust=0, mode='ser', core_count=1, store_on_gpu=False, smoothed_array=None):
    """
    This function serves as a handler for the different implementations of the IUWT recomposition. It allows the
    different methods to be used almost interchangeably.

    INPUTS:
    in1                 (no default):       Array on which the decomposition is to be performed.
    scale_adjust        (no default):       Number of omitted scales.
    mode                (default='ser')     Implementation of the IUWT to be used - 'ser', 'mp' or 'gpu'.
    core_count          (default=1)         Additional option for multiprocessing - specifies core count.
    store_on_gpu        (default=False):    Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    Returns the recomposition.
    """

    if mode=='ser':
        return ser_iuwt_recomposition(in1, scale_adjust, smoothed_array)
    elif mode=='mp':
        return mp_iuwt_recomposition(in1, scale_adjust, core_count, smoothed_array)
    elif mode=='gpu':
        return gpu_iuwt_recomposition(in1, scale_adjust, store_on_gpu, smoothed_array)

def ser_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for a single CPU core.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.

    OUTPUTS:
    detail_coeffs                       Array containing the detail coefficients.
    C0                  (optional):     Array containing the smoothest version of the input.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    # Initialises an empty array to store the coefficients.

    detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    C0 = in1    # Sets the initial value to be the input array.

    # The following loop, which iterates up to scale_adjust, applies the a trous algorithm to the scales which are
    # considered insignificant. This is important as each set of wavelet coefficients depends on the last smoothed
    # version of the input.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C0 = ser_a_trous(C0, wavelet_filter, i)

    # The meat of the algorithm - two sequential applications fo the a trous followed by determination and storing of
    # the detail coefficients. C0 is reassigned the value of C on each loop - C0 is always the smoothest version of the
    # input image.

    for i in range(scale_adjust,scale_count):
        C = ser_a_trous(C0, wavelet_filter, i)                                  # Approximation coefficients.
        C1 = ser_a_trous(C, wavelet_filter, i)                                  # Approximation coefficients.
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                             # Detail coefficients.
        C0 = C

    if store_smoothed:
        return detail_coeffs, C0
    else:
        return detail_coeffs

def ser_iuwt_recomposition(in1, scale_adjust, smoothed_array):
    """
    This function calls the a trous algorithm code to recompose the input into a single array. This is the
    implementation of the isotropic undecimated wavelet transform recomposition for a single CPU core.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of truncated array pages.
    smoothed_array  (default=None): For a complete inverse transform, this must be the smoothest approximation.

    OUTPUTS:
    recomposition                   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    # Determines scale with adjustment and creates a zero array to store the output, unless smoothed_array is given.

    max_scale = in1.shape[0] + scale_adjust

    if smoothed_array is None:
        recomposition = np.zeros([in1.shape[1], in1.shape[2]])
    else:
        recomposition = smoothed_array

    # The following loops call the a trous algorithm code to recompose the input. The first loop assumes that there are
    # non-zero wavelet coefficients at scales above scale_adjust, while the second loop completes the recomposition
    # on the scales less than scale_adjust.

    for i in range(max_scale-1, scale_adjust-1, -1):
        recomposition = ser_a_trous(recomposition, wavelet_filter, i) + in1[i-scale_adjust,:,:]

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            recomposition = ser_a_trous(recomposition, wavelet_filter, i)

    return recomposition

def ser_a_trous(C0, filter, scale):
    """
    The following is a serial implementation of the a trous algorithm. Accepts the following parameters:

    INPUTS:
    filter      (no default):   The filter-bank which is applied to the components of the transform.
    C0          (no default):   The current array on which filtering is to be performed.
    scale       (no default):   The scale for which the decomposition is being carried out.

    OUTPUTS:
    C1                          The result of applying the a trous algorithm to the input.
    """
    tmp = filter[2]*C0

    tmp[(2**(scale+1)):,:] += filter[0]*C0[:-(2**(scale+1)),:]
    tmp[:(2**(scale+1)),:] += filter[0]*C0[(2**(scale+1))-1::-1,:]

    tmp[(2**scale):,:] += filter[1]*C0[:-(2**scale),:]
    tmp[:(2**scale),:] += filter[1]*C0[(2**scale)-1::-1,:]

    tmp[:-(2**scale),:] += filter[3]*C0[(2**scale):,:]
    tmp[-(2**scale):,:] += filter[3]*C0[:-(2**scale)-1:-1,:]

    tmp[:-(2**(scale+1)),:] += filter[4]*C0[(2**(scale+1)):,:]
    tmp[-(2**(scale+1)):,:] += filter[4]*C0[:-(2**(scale+1))-1:-1,:]

    C1 = filter[2]*tmp

    C1[:,(2**(scale+1)):] += filter[0]*tmp[:,:-(2**(scale+1))]
    C1[:,:(2**(scale+1))] += filter[0]*tmp[:,(2**(scale+1))-1::-1]

    C1[:,(2**scale):] += filter[1]*tmp[:,:-(2**scale)]
    C1[:,:(2**scale)] += filter[1]*tmp[:,(2**scale)-1::-1]

    C1[:,:-(2**scale)] += filter[3]*tmp[:,(2**scale):]
    C1[:,-(2**scale):] += filter[3]*tmp[:,:-(2**scale)-1:-1]

    C1[:,:-(2**(scale+1))] += filter[4]*tmp[:,(2**(scale+1)):]
    C1[:,-(2**(scale+1)):] += filter[4]*tmp[:,:-(2**(scale+1))-1:-1]

    return C1

def mp_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, core_count):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for multiple CPU cores. NOTE: Python is not well suited
    to multiprocessing - this may not improve execution speed.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.
    core_count          (no default):   Indicates the number of cores to be used.

    OUTPUTS:
    detail_coeffs                       Array containing the detail coefficients.
    C0                  (optional):     Array containing the smoothest version of the input.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    C0 = in1                                            # Sets the initial value to be the input array.

    # Initialises a zero array to store the coefficients.

    detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    # The following loop, which iterates up to scale_adjust, applies the a trous algorithm to the scales which are
    # considered insignificant. This is important as each set of wavelet coefficients depends on the last smoothed
    # version of the input.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C0 = mp_a_trous(C0, wavelet_filter, i, core_count)

    # The meat of the algorithm - two sequential applications fo the a trous followed by determination and storing of
    # the detail coefficients. C0 is reassigned the value of C on each loop - C0 is always the smoothest version of the
    # input image.

    for i in range(scale_adjust,scale_count):
        C = mp_a_trous(C0, wavelet_filter, i, core_count)                   # Approximation coefficients.
        C1 = mp_a_trous(C, wavelet_filter, i, core_count)                   # Approximation coefficients.
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                         # Detail coefficients.
        C0 = C

    if store_smoothed:
        return detail_coeffs, C0
    else:
        return detail_coeffs

def mp_iuwt_recomposition(in1, scale_adjust, core_count, smoothed_array):
    """
    This function calls the a trous algorithm code to recompose the input into a single array. This is the
    implementation of the isotropic undecimated wavelet transform recomposition for multiple CPU cores.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of omitted array pages.
    core_count      (no default):   Indicates the number of cores to be used.
    smoothed_array  (default=None): For a complete inverse transform, this must be the smoothest approximation.

    OUTPUTS:
    recomposiiton                   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    # Determines scale with adjustment and creates a zero array to store the output, unless smoothed_array is given.

    max_scale = in1.shape[0] + scale_adjust

    if smoothed_array is None:
        recomposition = np.zeros([in1.shape[1], in1.shape[2]])
    else:
        recomposition = smoothed_array

    # The following loops call the a trous algorithm code to recompose the input. The first loop assumes that there are
    # non-zero wavelet coefficients at scales above scale_adjust, while the second loop completes the recomposition
    # on the scales less than scale_adjust.

    for i in range(max_scale-1, scale_adjust-1, -1):
        recomposition = mp_a_trous(recomposition, wavelet_filter, i, core_count) + in1[i-scale_adjust,:,:]

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            recomposition = mp_a_trous(recomposition, wavelet_filter, i, core_count)

    return recomposition

def mp_a_trous(C0, wavelet_filter, scale, core_count):
    """
    This is a reimplementation of the a trous filter which makes use of multiprocessing. In particular,
    it divides the input array of dimensions NxN into M smaller arrays of dimensions (N/M)xN, where M is the
    number of cores which are to be used.

    INPUTS:
    C0              (no default):   The current array which is to be decomposed.
    wavelet_filter  (no default):   The filter-bank which is applied to the components of the transform.
    scale           (no default):   The scale at which decomposition is to be carried out.
    core_count      (no default):   The number of CPU cores over which the task should be divided.

    OUTPUTS:
    shared_array                    The decomposed array.
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
    wavelet_filter  (no default):       The filter-bank which is applied to the elements of the transform.
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

def gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, store_on_gpu):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for a GPU.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (no default):   Adjustment to scale value if first scales are of no interest.
    store_smoothed      (no default):   Boolean specifier for whether the smoothed image is stored or not.
    store_on_gpu        (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    detail_coeffs                       Array containing the detail coefficients.
    C0                  (optional):     Array containing the smoothest version of the input.
    """

    # The following simple kernel just allows for the construction of a 3D decomposition on the GPU.

    ker = SourceModule("""
                        __global__ void gpu_store_detail_coeffs(float *in1, float *in2, float* out1, int *scale, int *adjust)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int j = (blockDim.y * blockIdx.y + threadIdx.y)*len;
                            const int k = (blockDim.z * blockIdx.z + threadIdx.z)*(len*len);
                            const int tid2 = i + j;
                            const int tid3 = i + j + k;

                            if ((blockIdx.z + adjust[0])==scale[0])
                                { out1[tid3] = in1[tid2] - in2[tid2]; }

                        }
                       """)

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)    # Filter-bank for use in the a trous algorithm.
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    # Initialises an empty array to store the detail coefficients.

    detail_coeffs = gpuarray.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]], np.float32)

    # Determines whether the array is already on the GPU or not. If not, moves it to the GPU.

    try:
        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))
    except:
        gpu_in1 = in1

    # Sets up some working arrays on the GPU to prevent memory transfers.

    gpu_tmp = gpuarray.empty_like(gpu_in1)
    gpu_out1 = gpuarray.empty_like(gpu_in1)
    gpu_out2 = gpuarray.empty_like(gpu_in1)

    # Sets up some parameters required by the algorithm on the GPU.

    gpu_scale = gpuarray.zeros([1], np.int32)
    gpu_adjust = gpuarray.zeros([1], np.int32)
    gpu_adjust += scale_adjust

    # Fetches the a trous kernels and sets up the unique storing kernel.

    gpu_a_trous_row_kernel, gpu_a_trous_col_kernel = gpu_a_trous()
    gpu_store_detail_coeffs = ker.get_function("gpu_store_detail_coeffs")

    grid_rows = int(in1.shape[0]//32)
    grid_cols = int(in1.shape[1]//32)

    # The following loop, which iterates up to scale_adjust, applies the a trous algorithm to the scales which are
    # considered insignificant. This is important as each set of wavelet coefficients depends on the last smoothed
    # version of the input.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

            gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

            gpu_in1, gpu_out1 = gpu_out1, gpu_in1
            gpu_scale += 1

    # The meat of the algorithm - two sequential applications fo the a trous followed by determination and storing of
    # the detail coefficients. C0 is reassigned the value of C on each loop - C0 is always the smoothest version of the
    # input image.

    for i in range(scale_adjust, scale_count):

        gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

        gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))           # Approximation coefficients.

        gpu_a_trous_row_kernel(gpu_out1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

        gpu_a_trous_col_kernel(gpu_tmp, gpu_out2, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))           # Approximation coefficients.

        gpu_store_detail_coeffs(gpu_in1, gpu_out2, detail_coeffs, gpu_scale, gpu_adjust,
                                block=(32,32,1), grid=(grid_cols, grid_rows, int(scale_count))) # Detail coefficients.

        gpu_in1, gpu_out1 = gpu_out1, gpu_in1
        gpu_scale += 1

    # Return values depend on mode. NOTE: store_smoothed does not work if the result stays on the gpu.

    if store_on_gpu:
        return detail_coeffs
    elif store_smoothed:
        return detail_coeffs.get(), gpu_in1.get()
    else:
        return detail_coeffs.get()

def gpu_iuwt_recomposition(in1, scale_adjust, store_on_gpu, smoothed_array):
    """
    This function calls the a trous algorithm code to recompose the input into a single array. This is the
    implementation of the isotropic undecimated wavelet transform recomposition for a GPU.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of omitted array pages.
    store_on_gpu    (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    recomposiiton                   Array containing the reconstructed array.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)    # Filter-bank for use in the a trous algorithm.
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    # Determines scale with adjustment and creates a zero array on the GPU to store the output,unless smoothed_array
    # is given.

    max_scale = in1.shape[0] + scale_adjust

    if smoothed_array is None:
        recomposition = gpuarray.zeros([in1.shape[1], in1.shape[2]], np.float32)
    else:
        recomposition = gpuarray.to_gpu(smoothed_array.astype(np.float32))

    # Determines whether the array is already on the GPU or not. If not, moves it to the GPU.

    try:
        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))
    except:
        gpu_in1 = in1

    # Creates a working array on the GPU.

    gpu_tmp = gpuarray.empty_like(recomposition)

    # Creates and fills an array with the appropriate scale value.

    gpu_scale = gpuarray.zeros([1], np.int32)
    gpu_scale += max_scale-1

     # Fetches the a trous kernels.

    gpu_a_trous_row_kernel, gpu_a_trous_col_kernel = gpu_a_trous()

    grid_rows = int(in1.shape[1]//32)
    grid_cols = int(in1.shape[2]//32)

    # The following loops call the a trous algorithm code to recompose the input. The first loop assumes that there are
    # non-zero wavelet coefficients at scales above scale_adjust, while the second loop completes the recomposition
    # on the scales less than scale_adjust.

    for i in range(max_scale-1, scale_adjust-1, -1):
        gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

        gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

        recomposition = recomposition[:,:] + gpu_in1[i-scale_adjust,:,:]

        gpu_scale -= 1

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

            gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(grid_cols, grid_rows))

            gpu_scale -= 1

    # Return values depend on mode.

    if store_on_gpu:
        return recomposition
    else:
        return recomposition.get()

def gpu_a_trous():
    """
    Simple convenience function so that the a trous kernels can be easily accessed by any function.
    """

    ker1 = SourceModule("""
                        __global__ void gpu_a_trous_row_kernel(float *in1, float *in2, float *wfil, int *scale)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int col = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int i = col;
                            const int row = (blockDim.y * blockIdx.y + threadIdx.y);
                            const int j = row*len;
                            const int tid2 = i + j;
                            const int lstp = exp2(float(scale[0] + 1));
                            const int sstp = exp2(float(scale[0]));

                            in2[tid2] = wfil[2]*in1[tid2];

                            if (row < lstp)
                                { in2[tid2] += wfil[0]*in1[col + len*(lstp - row - 1)]; }
                            else
                                { in2[tid2] += wfil[0]*in1[tid2 - lstp*len]; }

                            if (row < sstp)
                                { in2[tid2] += wfil[1]*in1[col + len*(sstp - row - 1)]; }
                            else
                                { in2[tid2] += wfil[1]*in1[tid2 - sstp*len]; }

                            if (row >= (len - sstp))
                                { in2[tid2] += wfil[3]*in1[col + len*(2*len - row - sstp - 1)]; }
                            else
                                { in2[tid2] += wfil[3]*in1[tid2 + sstp*len]; }

                            if (row >= (len - lstp))
                                { in2[tid2] += wfil[4]*in1[col + len*(2*len - row - lstp - 1)]; }
                            else
                                { in2[tid2] += wfil[4]*in1[tid2 + lstp*len]; }
                        }
                        """)

    ker2 = SourceModule("""
                        __global__ void gpu_a_trous_col_kernel(float *in1, float *in2, float *wfil, int *scale)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int col = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int i = col;
                            const int row = (blockDim.y * blockIdx.y + threadIdx.y);
                            const int j = row*len;
                            const int tid2 = i + j;
                            const int lstp = exp2(float(scale[0] + 1));
                            const int sstp = exp2(float(scale[0]));

                            in2[tid2] = wfil[2]*in1[tid2];

                            if (col < lstp)
                                { in2[tid2] += wfil[0]*in1[j - col + lstp - 1]; }
                            else
                                { in2[tid2] += wfil[0]*in1[tid2 - lstp]; }

                            if (col < sstp)
                                { in2[tid2] += wfil[1]*in1[j - col + sstp - 1]; }
                            else
                                { in2[tid2] += wfil[1]*in1[tid2 - sstp]; }

                            if (col >= (len - sstp))
                                { in2[tid2] += wfil[3]*in1[j + 2*len - sstp - col - 1]; }
                            else
                                { in2[tid2] += wfil[3]*in1[tid2 + sstp]; }

                            if (col >= (len - lstp))
                                { in2[tid2] += wfil[4]*in1[j + 2*len - lstp - col - 1]; }
                            else
                                { in2[tid2] += wfil[4]*in1[tid2 + lstp]; }

                        }
                       """)

    return ker1.get_function("gpu_a_trous_row_kernel"), ker2.get_function("gpu_a_trous_col_kernel")