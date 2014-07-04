import numpy as np
import pyfits
import time

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
            fft_in1 = gpu_r2c_fft(fft_in1, store_on_gpu=True)
            fft_in2 = in2

            conv_in1_in2 = fft_in1*fft_in2

            conv_in1_in2 = gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True)

            out1_slice = tuple(slice(0.5*sz,1.5*sz) for sz in in1.shape)

            return np.fft.fftshift(conv_in1_in2)[out1_slice]

        elif conv_mode=="circular":
            fft_in1 = gpu_r2c_fft(in1, store_on_gpu=True)
            fft_in2 = in2

            conv_in1_in2 = fft_in1*fft_in2

            conv_in1_in2 = gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True)

            return np.fft.fftshift(conv_in1_in2)
    else:

        if conv_mode=="linear":
            fft_in1 = pad_array(in1)
            fft_in2 = in2

            out1_slice = tuple(slice(0.5*sz,1.5*sz) for sz in in1.shape)

            return np.fft.fftshift(np.fft.irfft2(fft_in2*np.fft.rfft2(fft_in1)))[out1_slice]

        elif conv_mode=="circular":
            return np.fft.fftshift(np.fft.irfft2(in2*np.fft.rfft2(in1)))

def gpu_r2c_fft(in1, is_gpuarray=False, store_on_gpu=False):
    """
    This function makes use of the scikits implementation of the FFT for GPUs to take the real to complex FFT.

    INPUTS:
    in1             (no default):       The array on which the FFT is to be performed.
    is_gpuarray     (default=True):     Boolean specifier for whether or not input is on the gpu.
    store_on_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.

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

    if store_on_gpu:
        return gpu_out1
    else:
        return gpu_out1.get()

def gpu_c2r_ifft(in1, is_gpuarray=False, store_on_gpu=False):
    """
    This function makes use of the scikits implementation of the FFT for GPUs to take the complex to real IFFT.

    INPUTS:
    in1             (no default):       The array on which the IFFT is to be performed.
    is_gpuarray     (default=True):     Boolean specifier for whether or not input is on the gpu.
    store_on_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.

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

    if store_on_gpu:
        return gpu_out1
    else:
        return gpu_out1.get()

def pad_array(in1):
    """
    Simple convenience function to pad arrays for linear convolution.

    INPUTS:
    in1     (no default):   Input array which is to be padded.

    OUTPUTS:
    out1    (no default):   Padded version of the input.
    """

    padded_size = 2*np.array(in1.shape)

    out1 = np.zeros([padded_size[0],padded_size[1]])
    out1[padded_size[0]/4:3*padded_size[0]/4,padded_size[1]/4:3*padded_size[1]/4] = in1

    return out1

if __name__ == "__main__":

    for i in range(1):

        np.random.seed(1)

        a = np.random.randn(4096,4096).astype(np.float32)



        img_hdu_list = pyfits.open("3C147.fits")

        psf_hdu_list = pyfits.open("3C147_PSF.fits")



        dirty_data = (img_hdu_list[0].data[0,0,:,:]).astype(np.float32)

        psf_data = (psf_hdu_list[0].data[0,0,:,:]).astype(np.float32)



        img_hdu_list.close()

        psf_hdu_list.close()



        a = np.zeros_like(dirty_data)
        a[0,0] = 1

        delta = psf_data



        fftdelta1 = gpu_r2c_fft(psf_data, store_on_gpu=True)

        fftdelta2 = pad_array(psf_data)

        fftdelta2 = gpu_r2c_fft(fftdelta2, store_on_gpu=True)



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

            e = fft_convolve(a, np.fft.rfft2(pad_array(delta)), use_gpu=False, conv_mode='linear')

            t += time.time() - t1



        print "CPU - LINEAR AVG TIME:", t/1

        print "CIRCULAR MAX ERROR:", np.max(np.abs(b-c))

        print "CIRCULAR AVG ERROR:", np.average(np.abs(b-c))

        print "LINEAR MAX ERROR:", np.max(np.abs(d-e))

        print "LINEAR AVG ERROR:", np.average(np.abs(d-e))

        print np.argmax(b), np.argmax(c), np.argmax(d), np.argmax(e)

        print b[0:3,0:3],c[0:3,0:3],d[0:3,0:3],e[0:3,0:3], delta[2048:2051,2048:2051]