import pyfits
import numpy as np
import math
import iuwt
import iuwt_convolution as conv
import iuwt_toolbox as tools
import pylab as pb
import argparse

import time

class FitsImage:
    """A class for the manipulation of .fits images - in particular for implementing deconvolution."""

    def __init__(self, image_name, psf_name):
        """
        Opens the original .fits images specified by imagename and psfname and stores their contents in appropriate
        variables for later use. Also initialises variables to store the sizes of the psf and dirty image as these
        quantities are used repeatedly.
        """

        self.image_name = image_name
        self.psf_name = psf_name

        self.img_hdu_list = pyfits.open("{}".format(self.image_name))
        self.psf_hdu_list = pyfits.open("{}".format(self.psf_name))

        self.dirty_data = (self.img_hdu_list[0].data[0,0,:,:]).astype(np.float32)
        self.psf_data = (self.psf_hdu_list[0].data[0,0,:,:]).astype(np.float32)

        self.img_hdu_list.close()
        self.psf_hdu_list.close()

        self.dirty_data_shape = self.dirty_data.shape
        self.psf_data_shape = self.psf_data.shape

    def moresane(self, subregion=None, scale_count=None, sigma_level=4, loop_gain=0.1, tolerance=0.7, accuracy=1e-6,
                 major_loop_miter=100, minor_loop_miter=30, decom_mode="ser", core_count=1, conv_device='cpu',
                 conv_mode='linear', extraction_mode='cpu'):
        """
        Primary method for wavelet analysis and subsequent deconvolution.

        INPUTS:
        subregion           (default=None):     Size, in pixels, of the central region to be analyzed and deconvolved.
        scale_count         (default=None):     Maximum scale to be considered - maximum scale considered during
                                                initialisation.
        sigma_level         (default=4)         Number of sigma at which thresholding is to be performed.
        loop_gain           (default=0.1):      Loop gain for the deconvolution.
        tolerance           (default=0.7):      Tolerance level for object extraction. Significant objects contain
                                                wavelet coefficients greater than the tolerance multiplied by the
                                                maximum wavelet coefficient in the scale under consideration.
        accuracy            (default=1e-6):     Threshold on the standard deviation of the residual noise. Exit main
                                                loop when this threshold is reached.
        major_loop_miter    (default=100):      Maximum number of iterations allowed in the major loop. Exit
                                                condition.
        minor_loop_miter    (default=30):       Maximum number of iterations allowed in the minor loop. Serves as an
                                                exit condition when the SNR is does not reach a maximum.
        decom_mode          (default='ser'):    Specifier for decomposition mode - serial, multiprocessing, or gpu.
        core_count          (default=1):        In the event that multiprocessing, specifies the number of cores.
        conv_device         (default='cpu'):    Specifier for device to be used - cpu or gpu.
        conv_mode           (default='linear'): Specifier for convolution mode - linear or circular.
        extraction_mode     (default='cpu'):    Specifier for mode to be used - cpu or gpu.

        """

        # If neither subregion nor scale_count is specified, the following handles the assignment of default values.
        # The default value for subregion is the whole image. The default value for scale_count is the log to the
        # base two of the image dimensions minus one.

        if subregion is None:
            subregion = self.dirty_data_shape[0]

        if (scale_count is None) or (scale_count>math.log(self.dirty_data_shape[0],2)-1):
            scale_count = int(math.log(self.dirty_data_shape[0],2)-1)
            print "Assuming maximum scale is {}.".format(scale_count)

        # The following creates arrays with dimensions equal to subregion and containing the values of the dirty
        # image and psf in their central subregions. TODO: Allow specification of deconvolution center.

        dirty_subregion = self.dirty_data[
                              (self.dirty_data_shape[0]/2-subregion/2):(self.dirty_data_shape[0]/2+subregion/2),
                              (self.dirty_data_shape[1]/2-subregion/2):(self.dirty_data_shape[1]/2+subregion/2)]
        psf_subregion = self.psf_data[
                              (self.psf_data_shape[0]/2-subregion/2):(self.psf_data_shape[0]/2+subregion/2),
                              (self.psf_data_shape[1]/2-subregion/2):(self.psf_data_shape[1]/2+subregion/2)]

        # The following pre-loads the gpu with the fft of both the full PSF and the subregion of interest. If usegpu
        # is false, this simply precomputes the fft of the PSF.

        if conv_device=="gpu":

            if conv_mode=="circular":
                psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion, is_gpuarray=False, store_on_gpu=True)
                if psf_subregion.shape==self.psf_data_shape:
                    psf_data_fft = psf_subregion_fft
                else:
                    psf_data_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)

            elif conv_mode=="linear":
                psf_subregion_fft = conv.pad_array(psf_subregion)
                psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
                if psf_subregion.shape==self.psf_data_shape:
                    psf_data_fft = psf_subregion_fft
                else:
                    psf_data_fft = conv.pad_array(self.psf_data)
                    psf_data_fft = conv.gpu_r2c_fft(psf_data_fft, is_gpuarray=False, store_on_gpu=True)

            else:
                print "Invalid convolution mode - aborting execution."

        elif conv_device=="cpu":

            if conv_mode=="circular":
                psf_subregion_fft = np.fft.rfft2(psf_subregion)
                if psf_subregion.shape==self.psf_data_shape:
                    psf_data_fft = psf_subregion_fft
                else:
                    psf_data_fft = np.fft.rfft2(self.psf_data)

            elif conv_mode=="linear":
                psf_subregion_fft = conv.pad_array(psf_subregion)
                psf_subregion_fft = np.fft.rfft2(psf_subregion_fft)
                if psf_subregion.shape==self.psf_data_shape:
                    psf_data_fft = psf_subregion_fft
                else:
                    psf_data_fft = conv.pad_array(self.psf_data)
                    psf_data_fft = np.fft.rfft2(psf_data_fft)

        else:
            print "Invalid convolution device - aborting execution."

        # The following is a call to the first of the IUWT (Isotropic Undecimated Wavelet Transform) functions. This
        # returns the decomposition of the PSF. The norm of each scale is found - these correspond to the energies or
        # weighting factors which must be applied when locating maxima.

        psf_decomposition = iuwt.iuwt_decomposition(psf_subregion, scale_count, mode=decom_mode, core_count=core_count)

        psf_energies = np.empty([psf_decomposition.shape[0],1,1])

        for i in range(psf_energies.shape[0]):
            psf_energies[i] = np.sqrt(np.sum(np.square(psf_decomposition[i,:,:])))

######################################################MAJOR LOOP######################################################

        major_loop_niter = 0
        max_coeff = 1
        residual_std = 1

        # The following is the major loop. Its exit conditions are reached if if the number of major loop iterations
        # exceeds a user defined value, the maximum wavelet coefficient is zero or the standard deviation of the
        # residual drops below a user specified accuracy threshold.

        while (((major_loop_niter<major_loop_miter) & (max_coeff>0)) & (np.abs(residual_std)>accuracy)):

            stop_flag = 0   # A flag to break out of the following loop.
            min_scale = 0   # The current minimum scale of interest. If this ever equals or exceeds the scale_count
                            # value, it will also break the following loop.

            while ((stop_flag!=1) & (min_scale<scale_count)):

                # This is the IUWT decomposition of the dirty image subregion up to scale_count, followed by a
                # thresholding of the resulting wavelet coefficients based on the MAD estimator.

                dirty_decomposition = iuwt.iuwt_decomposition(dirty_subregion, scale_count, 0, decom_mode, core_count)
                dirty_decomposition_thresh = tools.threshold(dirty_decomposition, sigma_level=sigma_level)

                # The following calculates and stores the normalised maximum at each scale.

                normalised_scale_maxima = np.empty_like(psf_energies)

                for i in range(dirty_decomposition_thresh.shape[0]):
                    normalised_scale_maxima[i] = np.max(dirty_decomposition_thresh[i,:,:])/psf_energies[i]

                # The following stores the index, scale and value of the global maximum coefficient.

                max_index = np.argmax(normalised_scale_maxima)
                max_scale = max_index + 1
                max_coeff = normalised_scale_maxima[max_index,0,0]

                print "Maximum scale: ", max_scale
                print "Maximum coefficient: ", max_coeff

                # The following constitutes a major change to the original implementation - the aim is to establish
                # as soon as possible which scales are to be omitted on the current iteration. This attempts to find
                # a local maxima or empty scales below the maximum scale. If either is found, that scale all those
                # below it are ignored.

                scale_adjust = 0

                for i in range(max_index-1,-1,-1):
                    if max_index > 1:
                        if (normalised_scale_maxima[i,0,0] > normalised_scale_maxima[i+1,0,0]):
                            scale_adjust = i + 1
                            print "Scale {} contains a local maxima. Ignoring scales <= {}".format(scale_adjust, scale_adjust)
                            break
                    if (normalised_scale_maxima[i,0,0] == 0):
                        scale_adjust = i + 1
                        print "Scale {} is empty. Ignoring scales <= {}".format(scale_adjust, scale_adjust)
                        break

                # This is an escape condition for the loop. If the maximum coefficient is zero, then there is no
                # useful information left in the wavelets and MORESANE is complete.

                if max_coeff == 0:
                    print "No significant wavelet coefficients detected."
                    break
                    # stop_flag = 1
                    # continue

                # We choose to only consider scales up to the scale containing the maximum wavelet coefficient,
                # and ignore scales at or below the scale adjustment.

                dirty_decomposition_thresh = dirty_decomposition_thresh[scale_adjust:max_scale,:,:]

                # The following is a call to the externally defined source extraction function. It returns an array
                # populated with the wavelet coefficients of structures of interest in the image. This basically refers
                # to objects containing a maximum wavelet coefficient within some user-specified tolerance of the
                # maximum  at that scale.

                extracted_sources, extracted_sources_mask = \
                    tools.source_extraction(dirty_decomposition_thresh, tolerance, mode=extraction_mode)

                # The wavelet coefficients of the extracted sources are recomposed into a single image,
                # which should contain only the structures of interest.

                recomposed_sources = \
                    iuwt.iuwt_recomposition(extracted_sources, scale_adjust, decom_mode, core_count)

######################################################MINOR LOOP######################################################

                r = recomposed_sources.copy()
                p = recomposed_sources.copy()
                minor_loop_niter = 0

                while (minor_loop_niter<minor_loop_miter):

                    alpha_denominator = conv.fft_convolve(p, psf_data_fft, conv_device, conv_mode)
                    alpha_denominator = iuwt.iuwt_decomposition(alpha_denominator, scale_count, scale_adjust,
                                                                decom_mode, core_count)
                    alpha_denominator = extracted_sources_mask*alpha_denominator
                    alpha_denominator = iuwt.iuwt_recomposition(alpha_denominator, scale_adjust, decom_mode, core_count)
                    alpha_denominator = np.dot(p.reshape(1,-1),alpha_denominator.reshape(-1,1))[0,0]

                    alpha_numerator = np.dot(r.reshape(1,-1),r.reshape(-1,1))[0,0]

                    alpha = alpha_numerator/alpha_denominator

                    print alpha

                    break

                break
            break













if __name__ == "__main__":
    test = FitsImage("3C147.fits","3C147_PSF.fits")

    start_time = time.time()
    test.moresane(scale_count=1, tolerance=0.5, conv_mode="linear")
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
