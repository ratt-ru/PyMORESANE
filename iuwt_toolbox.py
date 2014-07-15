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

import pyfits
import iuwt
import pylab as pb
import time

def threshold(in1, sigma_level=4):
    """
    This function performs the thresholding of the values in array in1 based on the estimated standard deviation
    given by the MAD (median absolute deviation) estimator about zero.

    INPUTS:
    in1             (no default):   The array which is to be thresholded.
    sigma_level     (no default):   The number of estimated deviations at which thresholding is to occur.

    OUTPUTS:
    out1            (no default):   An thresholded version of in1.
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

def snr_ratio(in1, in2):
    """
    The following function simply calculates the signal to noise ratio between two signals.

    INPUTS:
    in1         (no default):   Array containing values for signal 1.
    in2         (no default):   Array containing values for signal 2.

    OUTPUTS:
    out1        (no default):   The ratio of the signal to noise ratios of two signals.
    """

    out1 = 20*np.log10(np.linalg.norm(in1)/np.linalg.norm(in1-in2))

    return out1

def cpu_source_extraction(in1, tolerance):

    scale_maxima = np.empty([in1.shape[0],1])
    objects = np.empty_like(in1, dtype=int)
    object_count = np.empty([in1.shape[0],1], dtype=int)

    for i in range(in1.shape[0]):
        scale_maxima[i] = np.max(in1[i,:,:])
        objects[i,:,:], object_count[i] = ndimage.label(in1[i,:,:], structure=[[1,1,1],[1,1,1],[1,1,1]])

    for i in range(-1,-in1.shape[0]-1,-1):
        if i==(-1):
            tmp = (in1[i,:,:]>(tolerance*scale_maxima[i]))*objects[i,:,:]
        else:
            tmp = (in1[i,:,:]>(tolerance*scale_maxima[i]))*objects[i,:,:]*objects[i+1,:,:]
        labels = np.unique(tmp[tmp>0])

        for j in labels:
            objects[i,(objects[i,:,:]==j)] = -1

        objects[i,(objects[i,:,:]>0)] = 0
        objects[i,:,:] = -(objects[i,:,:])

    return objects*in1

def gpu_source_extraction(in1, tolerance):

    ker1 = SourceModule("""
                        __global__ void gpu_mask_kernel1(int *in1, int *in2)
                        {
                          const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                          const int j = (blockDim.y * blockIdx.y + threadIdx.y)*gridDim.y*blockDim.y;
                          const int tid = i + j;

                          if (in1[tid] == in2[0])
                            {
                                in1[tid] = -1;
                            }
                        }
                       """)

    ker2 = SourceModule("""
                        __global__ void gpu_mask_kernel2(int *in1)
                        {
                          const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                          const int j = (blockDim.y * blockIdx.y + threadIdx.y)*gridDim.y*blockDim.y;
                          const int tid = i + j;

                          if (in1[tid] >= 0)
                            {
                                in1[tid] = 0;
                            }
                          else
                            {
                                in1[tid] = 1;
                            }
                        }
                       """)

    ker3 = SourceModule("""
                    __global__ void gpu_mask_kernel3(float *in1, float *in2)
                    {
                      const int i = (blockDim.x * blockIdx.x + threadIdx.x);
                      const int j = (blockDim.y * blockIdx.y + threadIdx.y)*gridDim.y*blockDim.y;
                      const int tid = i + j;

                      if (in1[tid] <= in2[0])
                        {
                            in1[tid] = 0;
                        }
                      else
                        {
                            in1[tid] = 1;
                        }
                    }
                   """)

    gpu_mask_kernel1 = ker1.get_function("gpu_mask_kernel1")
    gpu_mask_kernel2 = ker2.get_function("gpu_mask_kernel2")
    gpu_mask_kernel3 = ker3.get_function("gpu_mask_kernel3")

    scale_maxima = np.empty([in1.shape[0],1])
    objects = np.empty_like(in1, dtype=int)
    object_count = np.empty([in1.shape[0],1], dtype=int)

    for i in range(in1.shape[0]):
        scale_maxima[i] = np.max(in1[i,:,:])
        objects[i,:,:], object_count[i] = ndimage.label(in1[i,:,:], structure=[[1,1,1],[1,1,1],[1,1,1]])

    for i in range(-1,-in1.shape[0]-1,-1):

        gpu_objects = gpuarray.to_gpu_async(objects[i,:,:].astype(np.int32))
        gpu_in1 = gpuarray.to_gpu_async(in1[i,:,:].astype(np.float32))
        gpu_condition = gpuarray.to_gpu_async((tolerance*scale_maxima[i]).astype(np.float32))

        if i==(-1):
            gpu_mask_kernel3(gpu_in1, gpu_condition, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))
            tmp = (gpu_in1*gpu_objects).get()
        else:
            gpu_mask_kernel3(gpu_in1, gpu_condition, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))
            tmp = (gpu_in1*gpu_objects*gpu_objects_old).get()

        labels = np.unique(tmp[tmp>0]).astype(np.int32)
        print labels

        for j in labels:
            label = gpuarray.to_gpu_async(j)
            gpu_mask_kernel1(gpu_objects, label, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))

        gpu_mask_kernel2(gpu_objects, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))

        gpu_objects_old = gpu_objects.copy()
        objects[i,:,:] = gpu_objects.get()

    return objects*in1

if __name__=="__main__":
    img_hdu_list = pyfits.open("3C147.fits")
    psf_hdu_list = pyfits.open("3C147_PSF.fits")

    dirty_data = (img_hdu_list[0].data[0,0,:,:]).astype(np.float32)
    psf_data = (psf_hdu_list[0].data[0,0,:,:]).astype(np.float32)

    img_hdu_list.close()
    psf_hdu_list.close()

    decomposition = iuwt.iuwt_decomposition(dirty_data,3)
    print decomposition.shape

    t = time.time()
    thresh_decom = threshold(decomposition, 5)
    print time.time() - t

    t = time.time()
    extraction1 = cpu_source_extraction(thresh_decom, 0.5)
    print time.time() - t

    t = time.time()
    extraction2 = gpu_source_extraction(thresh_decom, 0.5)
    print time.time() - t

    a = np.ones([3,4096,4096])
    a = gpuarray.to_gpu_async(a)

    print np.where(extraction1!=extraction2)
    # for i in range(extraction.shape[0]):
    #     pb.imshow(extraction[i,:,:])
    #     pb.show()
