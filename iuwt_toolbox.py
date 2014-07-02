import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import time
import pylab as pb

try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from scikits.cuda.fft import Plan
    from scikits.cuda.fft import fft
    from scikits.cuda.fft import ifft
except:
    print "Pycuda unavailable - GPU mode will fail."

def fft_convolve(in1, in2, use_gpu=False, conv_mode="linear"):
    """
    This function determines the convolution of two inputs using the FFT. Contains an implementation for both CPU
    and GPU.

    INPUTS:
    in1         (no default):           Array containing one set of data, possibly an image.
    in2         (no default):           Gpuarray containing the FFT of the PSF.
    use_gpu     (default = True):       Boolean parameter which allows specification of GPU use.
    conv_mode   (default = "linear")    Mode specifier for the convolution.
    """

    # NOTE: Circular convolution assumes a periodic repetition of the input. This can cause edge effects. Linear
    # convolution pads the input with zeros to avoid this problem but is consequently heavier on computation and
    # memory.

    if use_gpu:
        if conv_mode=="linear":
            fft_in1 = pad_array(in1)
            fft_in1 = gpu_r2c_fft(fft_in1, load_gpu=True)
            fft_in2 = in2

            conv_in1_in2 = fft_in1*fft_in2

            conv_in1_in2 = gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True)

            out1_slice = tuple(slice(0.5*sz,1.5*sz) for sz in in1.shape)

            return np.fft.fftshift(conv_in1_in2)[out1_slice]
        elif conv_mode=="circular":
            fft_in1 = gpu_r2c_fft(in1, load_gpu=True)
            fft_in2 = in2

            conv_in1_in2 = fft_in1*fft_in2

            conv_in1_in2 = gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True)

            return np.fft.fftshift(conv_in1_in2)
    else:
        if conv_mode=="linear":
            return fftconvolve(in1, np.rot90(in2,2), mode='same')
        elif conv_mode=="circular":
            return np.fft.fftshift(np.fft.irfft2(in2*np.fft.rfft2(in1)))

def gpu_r2c_fft(in1, is_gpuarray=False, load_gpu=False):
    """
    This function makes use of the scikits implementation of the FFT for GPUs to take the real to complex FFT.

    INPUTS:
    in1         (no default):       The array on which the FFT is to be performed.
    is_gpuarray (default=True):     Boolean specifier for whether or not input is on the gpu.
    load_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.

    OUTPUTS:
    gpu_out1        (no default):   The gpu array containing the result.
        OR
    gpu_out1.get()  (no default):   The result from the gpu array.
    """

    if is_gpuarray:
        gpu_in1 = in1
    else:
        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))

    output_size = np.array(in1.shape)
    output_size[1] = 0.5*output_size[1] + 1

    gpu_out1 = gpuarray.empty([output_size[0], output_size[1]], np.complex64)
    gpu_plan = Plan(gpu_in1.shape, np.float32, np.complex64)
    fft(gpu_in1, gpu_out1, gpu_plan)

    if load_gpu:
        return gpu_out1
    else:
        return gpu_out1.get()

def gpu_c2r_ifft(in1, is_gpuarray=False, load_gpu=False):
    """
    This function makes use of the scikits implementation of the FFT for GPUs to take the complex to real IFFT.

    INPUTS:
    in1         (no default):       The array on which the IFFT is to be performed.
    is_gpuarray (default=True):     Boolean specifier for whether or not input is on the gpu.
    load_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.

    OUTPUTS:
    gpu_out1        (no default):   The gpu array containing the result.
        OR
    gpu_out1.get()  (no default):   The result from the gpu array.
    """

    if is_gpuarray:
        gpu_in1 = in1
    else:
        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.complex64))

    output_size = np.array(in1.shape)
    output_size[1] = 2*(output_size[1]-1)

    gpu_out1 = gpuarray.empty([output_size[0],output_size[1]], np.float32)
    gpu_plan = Plan(output_size, np.complex64, np.float32)
    ifft(gpu_in1, gpu_out1, gpu_plan, True)

    if load_gpu:
        return gpu_out1
    else:
        return gpu_out1.get()

def pad_array(in1):
    """
    Simple convenience function to pad arrays for linear convolution.

    INPUTS:
    in1     (no default):   Input array which is to be padded.

    OUTPUTS:
    out1    (no default):   Padded version fo the input.
    """

    padded_size = 2*np.array(in1.shape)

    out1 = np.zeros([padded_size[0],padded_size[1]])
    out1[padded_size[0]/4:3*padded_size[0]/4,padded_size[1]/4:3*padded_size[1]/4] = in1

    return out1

if __name__ == "__main__":
    for i in range(1):
        np.random.seed(1)
        a = np.random.randn(4096,4096).astype(np.float32)

        delta = np.zeros_like(a)
        delta[delta.shape[0]/2,delta.shape[1]/2] = 1

        fftdelta1 = gpu_r2c_fft(delta.astype(np.float32), load_gpu=True)
        fftdelta2 = pad_array(delta)
        fftdelta2 = gpu_r2c_fft(fftdelta2, load_gpu=True)

        t = 0

        for i in range(10):
            t1 = time.time()
            b = fft_convolve(a, fftdelta1, use_gpu=True, conv_mode='circular')
            t += time.time() - t1

        print "GPU - CIRCULAR AVG TIME:", t/10

        t = 0

        for i in range(5):
            t1 = time.time()
            c = fft_convolve(a, np.fft.rfft2(delta), use_gpu=False, conv_mode='circular')
            t += time.time() - t1

        print "CPU - CIRCULAR AVG TIME:", t/5

        t = 0

        for i in range(10):
            t1 = time.time()
            d = fft_convolve(a, fftdelta2, use_gpu=True, conv_mode='linear')
            t += time.time() - t1

        print "GPU - LINEAR AVG TIME:", t/10

        t = 0

        for i in range(1):
            t1 = time.time()
            e = fft_convolve(a, delta, use_gpu=False, conv_mode='linear')
            t += time.time() - t1

        print "CPU - LINEAR AVG TIME:", t/1

        print "GPU - CIRCULAR MAX ERROR:", np.max(np.abs(b-a))
        print "GPU - CIRCULAR AVG ERROR:", np.average(np.abs(b-a))
        print "CPU - CIRCULAR MAX ERROR:", np.max(np.abs(c-a))
        print "CPU - CIRCULAR AVG ERROR:", np.average(np.abs(c-a))
        print "GPU - LINEAR MAX ERROR:", np.max(np.abs(d-a))
        print "GPU - LINEAR AVG ERROR:", np.average(np.abs(d-a))
        print "CPU - LINEAR MAX ERROR:", np.max(np.abs(e-a))
        print "CPU - LINEAR MAX ERROR:", np.average(np.abs(e-a))

# def threshold_array(in1, max_scale, initial_run=False):
#     """
#     This function performs the thresholding of the values in in1 at various sigma levels. When
#     initialrun is True, thresholding will be performed at 5 sigma uniformly. When it is False, thresholding is
#     performed at 3 sigma for scales less than maxscale. An additional 3 sigma threshold is also returned for maxscale.
#     Accepts the following paramters:
#
#     in1                     (no default):   The array to which the threshold is to be applied.
#     max_scale               (no default):   The maximum scale of of the decomposition.
#     initialrun              (default=False):A boolean which determines whether thresholding is at 3 or 5 sigma.
#     """
#
#     # The following establishes which thresholding level is of interest.
#
#     if initial_run:
#         sigma_level = 5
#     else:
#         sigma_level = 3
#
#     # The following loop iterates up to maxscale, and calculates the thresholded components using the functionality
#     # of np.where to create masks at each scale. Components of interest are determined by whether they are within
#     # some sigma of the threshold level, which is defined to be the median of the absolute values of the current
#     # scale divided by some factor, 0.6754.
#
#     for i in range(max_scale):
#         threshold_level = (np.median(np.abs(in1[i,:,:]))/0.6754)
#
#         if (i==(max_scale-1))&~initial_run:
#             mask = np.where((np.abs(in1[i,:,:])<(sigma_level*threshold_level)),0,1)
#             thresh3sigma = (mask*in1[i,:,:] + np.abs(mask*in1[i,:,:]))/2
#             sigma_level = 5
#
#         mask = np.where((np.abs(in1[i,:,:])<(sigma_level*threshold_level)),0,1)
#         in1[i,:,:] = (mask*in1[i,:,:] + np.abs(mask*in1[i,:,:]))/2
#
#     # The following simply determines which values to return.
#
#     if initial_run:
#         return in1
#     else:
#         return in1, thresh3sigma