import numpy as np
from scipy import ndimage

try:
    import pycuda.driver as drv
    import pycuda.tools
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
except:
    print "Pycuda unavailable - GPU mode will fail."

def threshold(in1, sigma_level=4):
    """
    This function performs the thresholding of the values in array in1 based on the estimated standard deviation
    given by the MAD (median absolute deviation) estimator about zero.

    INPUTS:
    in1             (no default):   The array which is to be thresholded.
    sigma_level     (no default):   The number of estimated deviations at which thresholding is to occur.

    OUTPUTS:
    out1                            An thresholded version of in1.
    """

    out1 = np.empty_like(in1)

    # The conditional here ensures that the function works even when only one scale is considered. Both cases are the
    # same: the MAD estimator is calculated and then the resulting value is used to threshold the input. NOTE: This
    # discards all negative coefficients.

    if len(in1.shape)==2:
        threshold_level = np.median(np.abs(in1))/0.6745                     # MAD estimator for normal distribution.
        out1 = (in1>(sigma_level*threshold_level))*in1
    else:
        for i in range(in1.shape[0]):
            threshold_level = np.median(np.abs(in1[i,:,:]))/0.6745          # MAD estimator for normal distribution.
            out1[i,:,:] = (in1[i,:,:]>(sigma_level*threshold_level))*in1[i,:,:]

    return out1

def source_extraction(in1, tolerance, mode="cpu", store_on_gpu=False):
    """
    Convenience function for allocating work to cpu or gpu, depending on the selected mode.

    INPUTS:
    in1         (no default):   Array containing the wavelet decomposition.
    tolerance   (no default):   Percentage of maximum coefficient at which objects are deemed significant.
    mode        (default="cpu"):Mode of operation - either "gpu" or "cpu".

    OUTPUTS:
    Array containing the significant wavelet coefficients of extracted sources.
    """

    if mode=="cpu":
        return cpu_source_extraction(in1, tolerance)
    elif mode=="gpu":
        return gpu_source_extraction(in1, tolerance, store_on_gpu)

def cpu_source_extraction(in1, tolerance):
    """
    The following function determines connectivity within a given wavelet decomposition. These connected and labelled
    structures are thresholded to within some tolerance of the maximum coefficient at the scale. This determines
    whether on not an object is to be considered as significant. Significant objects are extracted and factored into
    a mask which is finally multiplied by the wavelet coefficients to return only wavelet coefficients belonging to
    significant objects across all scales.

    INPUTS:
    in1         (no default):   Array containing the wavelet decomposition.
    tolerance   (no default):   Percentage of maximum coefficient at which objects are deemed significant.

    OUTPUTS:
    objects*in1                 The wavelet coefficients of the significant structures.
    objects                     The mask of the significant structures.
    """

    # The following initialises some variables for storing the labelled image and the number of labels. The per scale
    # maxima are also initialised here.

    scale_maxima = np.empty([in1.shape[0],1])

    objects = np.empty_like(in1, dtype=np.int32)
    object_count = np.empty([in1.shape[0],1], dtype=np.int32)

    # The following loop uses functionality from the ndimage module to assess connectivity. The maxima are also
    # calculated here.

    for i in range(in1.shape[0]):
        scale_maxima[i] = np.max(in1[i,:,:])
        objects[i,:,:], object_count[i] = ndimage.label(in1[i,:,:], structure=[[1,1,1],[1,1,1],[1,1,1]])

    # The following removes the insignificant objects and then extracts the remaining ones.

    for i in range(-1,-in1.shape[0]-1,-1):
        if i==(-1):
            tmp = (in1[i,:,:]>=(tolerance*scale_maxima[i]))*objects[i,:,:]
        else:
            tmp = (in1[i,:,:]>=(tolerance*scale_maxima[i]))*objects[i,:,:]*objects[i+1,:,:]

        labels = np.unique(tmp[tmp>0])

        for j in labels:
            objects[i,(objects[i,:,:]==j)] = -1

        objects[i,(objects[i,:,:]>0)] = 0
        objects[i,:,:] = -(objects[i,:,:])

    return objects*in1, objects

def gpu_source_extraction(in1, tolerance, store_on_gpu):
    """
    The following function determines connectivity within a given wavelet decomposition. These connected and labelled
    structures are thresholded to within some tolerance of the maximum coefficient at the scale. This determines
    whether on not an object is to be considered as significant. Significant objects are extracted and factored into
    a mask which is finally multiplied by the wavelet coefficients to return only wavelet coefficients belonging to
    significant objects across all scales. This GPU accelerated version speeds up the extraction process.

    INPUTS:
    in1         (no default):   Array containing the wavelet decomposition.
    tolerance   (no default):   Percentage of maximum coefficient at which objects are deemed significant.
    store_on_gpu(no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.

    OUTPUTS:
    objects*in1                 The wavelet coefficients of the significant structures.
    objects                     The mask of the significant structures - if store_on_gpu is True, returns a gpuarray.
    """

    # The following are pycuda kernels which are executed on the gpu. Specifically, these both perform thresholding
    # operations. The gpu is much faster at this on large arrays due to their massive parallel processing power.

    ker1 = SourceModule("""
                        __global__ void gpu_mask_kernel1(int *in1, int *in2)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int j = (blockDim.y * blockIdx.y + threadIdx.y)*len;
                            const int tid2 = i + j;

                            if (in1[tid2] == in2[0])
                                { in1[tid2] = -1; }
                        }
                       """)

    ker2 = SourceModule("""
                        __global__ void gpu_mask_kernel2(int *in1)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int j = (blockDim.y * blockIdx.y + threadIdx.y)*len;
                            const int tid2 = i + j;

                            if (in1[tid2] >= 0)
                                { in1[tid2] = 0; }
                            else
                                { in1[tid2] = 1; }
                        }
                       """)

    ker3 = SourceModule("""
                        __global__ void gpu_store_objects(int *in1, float *out1, int *scale)
                        {
                            const int len = gridDim.x*blockDim.x;
                            const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                            const int j = (blockDim.y * blockIdx.y + threadIdx.y)*len;
                            const int k = (blockDim.z * blockIdx.z + threadIdx.z)*(len*len);
                            const int tid2 = i + j;
                            const int tid3 = i + j + k;

                            if (blockIdx.z==scale[0])
                                { out1[tid3] = in1[tid2]; }
                        }
                       """)

    # The following initialises some variables for storing the labelled image and the number of labels. The per scale
    # maxima are also initialised here.

    scale_maxima = np.empty([in1.shape[0],1], dtype=np.float32)
    objects = np.empty_like(in1, dtype=np.int32)
    object_count = np.empty([in1.shape[0],1], dtype=np.int32)

    # The following loop uses functionality from the ndimage module to assess connectivity. The maxima are also
    # calculated here.

    for i in range(in1.shape[0]):
        scale_maxima[i] = np.max(in1[i,:,:])
        objects[i,:,:], object_count[i] = ndimage.label(in1[i,:,:], structure=[[1,1,1],[1,1,1],[1,1,1]])

    # The following bind the pycuda kernels to the expressions on the left.

    gpu_mask_kernel1 = ker1.get_function("gpu_mask_kernel1")
    gpu_mask_kernel2 = ker2.get_function("gpu_mask_kernel2")

    # If store_on_gpu is the following handles some initialisation.

    if store_on_gpu:
        gpu_store_objects = ker3.get_function("gpu_store_objects")
        gpu_objects = gpuarray.empty(objects.shape, np.float32)
        gpu_idx = gpuarray.zeros([1], np.int32)
        gpu_idx += (objects.shape[0]-1)

    # The following removes the insignificant objects and then extracts the remaining ones.

    for i in range(-1,-in1.shape[0]-1,-1):

        condition = tolerance*scale_maxima[i]

        if i==(-1):
            tmp = (in1[i,:,:]>=condition)*objects[i,:,:]
        else:
            tmp = (in1[i,:,:]>=condition)*objects[i,:,:]*objects[i+1,:,:]

        labels = (np.unique(tmp[tmp>0])).astype(np.int32)

        gpu_objects_page = gpuarray.to_gpu_async(objects[i,:,:].astype(np.int32))

        for j in labels:
            label = gpuarray.to_gpu_async(j)
            gpu_mask_kernel1(gpu_objects_page, label, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))

        gpu_mask_kernel2(gpu_objects_page, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))

        objects[i,:,:] = gpu_objects_page.get()

        # In the event that all operations are to be done on the GPU, the following stores a version of the objects
        # on the GPU. A handle to the gpuarray is then returned.

        if store_on_gpu:
            gpu_store_objects(gpu_objects_page, gpu_objects, gpu_idx, block=(32,32,1), grid=(objects.shape[2]//32,
                                objects.shape[1]//32, objects.shape[0]))
            gpu_idx -= 1

    if store_on_gpu:
        return objects*in1, gpu_objects
    else:
        return objects*in1, objects

def snr_ratio(in1, in2):
    """
    The following function simply calculates the signal to noise ratio between two signals.

    INPUTS:
    in1         (no default):   Array containing values for signal 1.
    in2         (no default):   Array containing values for signal 2.

    OUTPUTS:
    out1                        The ratio of the signal to noise ratios of two signals.
    """

    out1 = 20*(np.log10(np.linalg.norm(in1)/np.linalg.norm(in1-in2)))

    return out1