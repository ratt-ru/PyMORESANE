import numpy as np
import time
import iuwt

try:
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
except:
    print "Pycuda unavailable - GPU mode will fail."

def gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, store_on_gpu):
    """
    This function exploits pycuda to implement the IUWT decomposition using kernels executed on the GPU. This version
    uses the minimum number of memory copies to achieve a substantially accelerated implementation.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (no default):   Adjustment to scale value if first scales are of no interest.
    store_smoothed      (no default):   Boolean specifier for whether the smoothed image is stored or not.
    store_on_gpu        (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    detail_coeffs       (no default): Array containing the detail coefficients and the smoothed image.
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

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    # Initialises an empty array to store the coefficients.

    detail_coeffs = gpuarray.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]], np.float32)

    # Sets up some working arrays on the GPU to prevent memory transfers.

    try:
        gpu_in1 = gpuarray.to_gpu_async(in1)
    except:
        gpu_in1 = in1

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

    # Functionality for when scale_adjust is non-zero - omits detail coefficient calculation.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

            gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

            gpu_in1, gpu_out1 = gpu_out1, gpu_in1
            gpu_scale += 1

    # The meat of the algorithm - two sequential a trous decompositions followed by determination and storing of the
    # detail coefficients.

    for i in range(scale_adjust, scale_count):

        gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

        gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

        gpu_a_trous_row_kernel(gpu_out1, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

        gpu_a_trous_col_kernel(gpu_tmp, gpu_out2, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32))

        gpu_store_detail_coeffs(gpu_in1, gpu_out2, detail_coeffs, gpu_scale, gpu_adjust,
                                block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[0]//32, scale_count))

        gpu_in1, gpu_out1 = gpu_out1, gpu_in1
        gpu_scale += 1

    # Return values depend on mode. NOTE: store_smoothed does not work if the result stays on the gpu.

    if store_on_gpu:
        return detail_coeffs
    elif store_smoothed:
        return detail_coeffs.get(), gpu_out1.get()
    else:
        return detail_coeffs.get()

def gpu_iuwt_recomposition(in1, scale_adjust, store_on_gpu):
    """
    This function calls the a trous algorithm code to recompose the input into a single image. This is the
    GPU implementation of the isotropic undecimated wavelet transform recomposition.

    INPUTS:
    in1             (no default):   Array containing wavelet coefficients.
    scale_adjust    (no default):   Indicates the number of truncated array pages.
    store_on_gpu    (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    recomposiiton   (no default):   Array containing the reconstructed image.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1], dtype=np.float32)
    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)

    # Determines scale with adjustment and creates a zero array on the GPU for the answer.

    max_scale = in1.shape[0] + scale_adjust
    recomposition = gpuarray.zeros([in1.shape[1], in1.shape[2]], np.float32)

    # Determines whether the array is already on the GPU or not. If not, moves it to the GPU.

    try:
        gpu_in1 = gpuarray.to_gpu_async(in1)
    except:
        gpu_in1 = in1

    # Creates a working array on the GPU.

    gpu_tmp = gpuarray.empty_like(recomposition)

    # Creates and fills an array with the appropriate scale value.

    gpu_scale = gpuarray.zeros([1], np.int32)
    gpu_scale += max_scale-1

     # Fetches the a trous kernels.

    gpu_a_trous_row_kernel, gpu_a_trous_col_kernel = gpu_a_trous()

    # Performs the recomposition at the scale for which we have explicit information.

    for i in range(max_scale-1, scale_adjust-1, -1):
        gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[2]//32, in1.shape[1]//32))

        gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[2]//32, in1.shape[1]//32))

        recomposition = recomposition[:,:] + gpu_in1[i-scale_adjust,:,:]

        gpu_scale -= 1

    # In the case of non-zero scale_adjust, completes the recomposition on the omitted scales.

    if scale_adjust>0:
        for i in range(scale_adjust-1, -1, -1):
            gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[2]//32, in1.shape[1]//32))

            gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
                                block=(32,32,1), grid=(in1.shape[2]//32, in1.shape[1]//32))

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

if __name__=="__main__":
    test = np.random.randn(4096,4096).astype(np.float32)
    gpu_test = gpuarray.to_gpu(test)

    for i in range(5):
        t = time.time()
        example = iuwt.iuwt_decomposition(test, 5, 0, mode='gpu')
        example = iuwt.iuwt_recomposition(example, 0, mode='gpu')
        print time.time() - t

    for i in range(5):
        t = time.time()
        result = gpu_iuwt_decomposition(gpu_test, 5, 0, False, True)
        result = gpu_iuwt_recomposition(result, 0, False)
        print time.time() - t
        gpu_test = gpuarray.to_gpu(test)

    print np.allclose(result ,example)
    print np.where(result!=example)
