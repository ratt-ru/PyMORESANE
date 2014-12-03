#!/usr/bin/env python
import logging
import pyfits
import numpy as np
import iuwt
import iuwt_convolution as conv
import iuwt_toolbox as tools
import pymoresane_parser as pparser
import beam_fit
import time
import os,sys

class FitsImage:
    """A class for the manipulation of .fits images - in particular for implementing deconvolution."""

    def __init__(self, image_name, psf_name=None,imageData=None,psfData=None,imageHdr=None,psfHdr=None):
        """
        Opens the original .fits images specified by imagename and psfname and stores their contents in appropriate
        variables for later use. Also initialises variables to store the sizes of the psf and dirty image as these
        quantities are used repeatedly.

        INPUTS:
        image_name  (no default):   Name of the input .fits file containing the dirty map.
        psf_name    (no default):   Name of the input .fits file containing the PSF.
        """

        self.image_name = image_name
        self.psf_name = psf_name

        self.imageData = imageData
        self.psfData = psfData
        self.imageHdr = imageHdr
        self.psfHdr = psfHdr
        if None not in [self.imageData,self.psfData,self.imageHdr,self.psfHdr]:
          self.dirty_data = self.imageData
          self.img_hdr = self.imageHdr
          self.psf_data = self.psfData
          self.psf_hdr = self.psfHdr
        elif self.psf_name :
          self.img_hdu_list = pyfits.open("{}".format(self.image_name))
          self.psf_hdu_list = pyfits.open("{}".format(self.psf_name))

          self.img_hdr = self.img_hdu_list[0].header
          self.psf_hdr = self.psf_hdu_list[0].header

          img_slice = self.handle_input(self.img_hdr)
          psf_slice = self.handle_input(self.psf_hdr)

          self.dirty_data = (self.img_hdu_list[0].data[img_slice]).astype(np.float32)
          self.psf_data = (self.psf_hdu_list[0].data[psf_slice]).astype(np.float32)


          self.img_hdu_list.close()
          self.psf_hdu_list.close()
 
        self.dirty_data_shape = self.dirty_data.shape
        self.psf_data_shape = self.psf_data.shape

        self.complete = False
        self.model = np.zeros_like(self.dirty_data)
        self.residual = np.copy(self.dirty_data)
        self.restored = np.zeros_like(self.dirty_data)

    def moresane(self, subregion=None, scale_count=None, sigma_level=4, loop_gain=0.1, tolerance=0.75, accuracy=1e-6,
                 major_loop_miter=100, minor_loop_miter=30, all_on_gpu=False, decom_mode="ser", core_count=1,
                 conv_device='cpu', conv_mode='linear', extraction_mode='cpu', enforce_positivity=False,
                 edge_suppression=False, edge_offset=0):
        """
        Primary method for wavelet analysis and subsequent deconvolution.

        INPUTS:
        subregion           (default=None):     Size, in pixels, of the central region to be analyzed and deconvolved.
        scale_count         (default=None):     Maximum scale to be considered - maximum scale considered during
                                                initialisation.
        sigma_level         (default=4)         Number of sigma at which thresholding is to be performed.
        loop_gain           (default=0.1):      Loop gain for the deconvolution.
        tolerance           (default=0.75):     Tolerance level for object extraction. Significant objects contain
                                                wavelet coefficients greater than the tolerance multiplied by the
                                                maximum wavelet coefficient in the scale under consideration.
        accuracy            (default=1e-6):     Threshold on the standard deviation of the residual noise. Exit main
                                                loop when this threshold is reached.
        major_loop_miter    (default=100):      Maximum number of iterations allowed in the major loop. Exit
                                                condition.
        minor_loop_miter    (default=30):       Maximum number of iterations allowed in the minor loop. Serves as an
                                                exit condition when the SNR is does not reach a maximum.
        all_on_gpu          (default=False):    Boolean specifier to toggle all gpu modes on.
        decom_mode          (default='ser'):    Specifier for decomposition mode - serial, multiprocessing, or gpu.
        core_count          (default=1):        For multiprocessing, specifies the number of cores.
        conv_device         (default='cpu'):    Specifier for device to be used - cpu or gpu.
        conv_mode           (default='linear'): Specifier for convolution mode - linear or circular.
        extraction_mode     (default='cpu'):    Specifier for mode to be used - cpu or gpu.
        enforce_positivity  (default=False):    Boolean specifier for whether or not a model must be strictly positive.
        edge_suppression    (default=False):    Boolean specifier for whether or not the edges are to be suprressed.
        edge_offset         (default=0):        Numeric value for an additional user-specified number of edge pixels
                                                to be ignored. This is added to the minimum suppression.

        OUTPUTS:
        self.model          (no default):       Model extracted by the algorithm.
        self.residual       (no default):       Residual signal after deconvolution.

        """

        # If neither subregion nor scale_count is specified, the following handles the assignment of default values.
        # The default value for subregion is the whole image. The default value for scale_count is the log to the
        # base two of the image dimensions minus one.

        logger.info("Starting...")

        if (subregion is None)|(subregion>self.dirty_data_shape[0]):
            subregion = self.dirty_data_shape[0]
            logger.info("Assuming subregion is {}px.".format(self.dirty_data_shape[0]))

        if (scale_count is None) or (scale_count>(np.log2(self.dirty_data_shape[0])-1)):
            scale_count = int(np.log2(self.dirty_data_shape[0])-1)
            logger.info("Assuming maximum scale is {}.".format(scale_count))

        if all_on_gpu:
            decom_mode = 'gpu'
            conv_device = 'gpu'
            extraction_mode = 'gpu'

        # The following creates arrays with dimensions equal to subregion and containing the values of the dirty
        # image and psf in their central subregions.

        subregion_slice = tuple([slice(self.dirty_data_shape[0]/2-subregion/2, self.dirty_data_shape[0]/2+subregion/2),
                                 slice(self.dirty_data_shape[1]/2-subregion/2, self.dirty_data_shape[1]/2+subregion/2)])

        dirty_subregion = self.dirty_data[subregion_slice]

        if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
            psf_subregion = self.psf_data[self.psf_data_shape[0]/2-subregion/2:self.psf_data_shape[0]/2+subregion/2,
                                          self.psf_data_shape[1]/2-subregion/2:self.psf_data_shape[1]/2+subregion/2]
        else:
            psf_subregion = self.psf_data[subregion_slice]

        # The following pre-loads the gpu with the fft of both the full PSF and the subregion of interest. If usegpu
        # is false, this simply precomputes the fft of the PSF.

        if conv_device=="gpu":
            if conv_mode=="circular":
                if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
                    psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion, is_gpuarray=False, store_on_gpu=True)
                    psf_slice = tuple([slice(self.psf_data_shape[0]/2-self.dirty_data_shape[0]/2, self.psf_data_shape[0]/2+self.dirty_data_shape[0]/2),
                                       slice(self.psf_data_shape[1]/2-self.dirty_data_shape[1]/2, self.psf_data_shape[1]/2+self.dirty_data_shape[1]/2)])
                    psf_data_fft = self.psf_data[psf_slice]
                    psf_data_fft = conv.gpu_r2c_fft(psf_data_fft, is_gpuarray=False, store_on_gpu=True)
                else:
                    psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion, is_gpuarray=False, store_on_gpu=True)
                    if psf_subregion.shape==self.psf_data_shape:
                        psf_data_fft = psf_subregion_fft
                    else:
                        psf_data_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)

            if conv_mode=="linear":
                if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
                    if np.all(np.array(self.dirty_data_shape)==subregion):
                        psf_subregion_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)
                        psf_data_fft = psf_subregion_fft
                        logger.info("Using double size PSF.")
                    else:
                        psf_slice = tuple([slice(self.psf_data_shape[0]/2-subregion, self.psf_data_shape[0]/2+subregion),
                                           slice(self.psf_data_shape[1]/2-subregion, self.psf_data_shape[1]/2+subregion)])
                        psf_subregion_fft = self.psf_data[psf_slice]
                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
                        psf_data_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)
                else:
                    if np.all(np.array(self.dirty_data_shape)==subregion):
                        psf_subregion_fft = conv.pad_array(self.psf_data)
                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
                        psf_data_fft = psf_subregion_fft
                    else:
                        psf_slice = tuple([slice(self.psf_data_shape[0]/2-subregion, self.psf_data_shape[0]/2+subregion),
                                           slice(self.psf_data_shape[1]/2-subregion, self.psf_data_shape[1]/2+subregion)])
                        psf_subregion_fft = self.psf_data[psf_slice]
                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
                        psf_data_fft = conv.pad_array(self.psf_data)
                        psf_data_fft = conv.gpu_r2c_fft(psf_data_fft, is_gpuarray=False, store_on_gpu=True)

        elif conv_device=="cpu":
            if conv_mode=="circular":
                if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
                    psf_subregion_fft = np.fft.rfft2(psf_subregion)
                    psf_slice = tuple([slice(self.psf_data_shape[0]/2-self.dirty_data_shape[0]/2, self.psf_data_shape[0]/2+self.dirty_data_shape[0]/2),
                                       slice(self.psf_data_shape[1]/2-self.dirty_data_shape[1]/2, self.psf_data_shape[1]/2+self.dirty_data_shape[1]/2)])
                    psf_data_fft = self.psf_data[psf_slice]
                    psf_data_fft = np.fft.rfft2(psf_data_fft)
                else:
                    psf_subregion_fft = np.fft.rfft2(psf_subregion)
                    if psf_subregion.shape==self.psf_data_shape:
                        psf_data_fft = psf_subregion_fft
                    else:
                        psf_data_fft = np.fft.rfft2(self.psf_data)

            if conv_mode=="linear":
                if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
                    if np.all(np.array(self.dirty_data_shape)==subregion):
                        psf_subregion_fft = np.fft.rfft2(self.psf_data)
                        psf_data_fft = psf_subregion_fft
                        logger.info("Using double size PSF.")
                    else:
                        psf_slice = tuple([slice(self.psf_data_shape[0]/2-subregion, self.psf_data_shape[0]/2+subregion),
                                           slice(self.psf_data_shape[1]/2-subregion, self.psf_data_shape[1]/2+subregion)])
                        psf_subregion_fft = self.psf_data[psf_slice]
                        psf_subregion_fft = np.fft.rfft2(psf_subregion_fft)
                        psf_data_fft = np.fft.rfft2(self.psf_data)
                else:
                    if np.all(np.array(self.dirty_data_shape)==subregion):
                        psf_subregion_fft = conv.pad_array(self.psf_data)
                        psf_subregion_fft = np.fft.rfft2(psf_subregion_fft)
                        psf_data_fft = psf_subregion_fft
                    else:
                        psf_slice = tuple([slice(self.psf_data_shape[0]/2-subregion, self.psf_data_shape[0]/2+subregion),
                                           slice(self.psf_data_shape[1]/2-subregion, self.psf_data_shape[1]/2+subregion)])
                        psf_subregion_fft = self.psf_data[psf_slice]
                        psf_subregion_fft = np.fft.rfft2(psf_subregion_fft)
                        psf_data_fft = conv.pad_array(self.psf_data)
                        psf_data_fft = np.fft.rfft2(psf_data_fft)

        # The following is a call to the first of the IUWT (Isotropic Undecimated Wavelet Transform) functions. This
        # returns the decomposition of the PSF. The norm of each scale is found - these correspond to the energies or
        # weighting factors which must be applied when locating maxima.

        psf_decomposition = iuwt.iuwt_decomposition(psf_subregion, scale_count, mode=decom_mode, core_count=core_count)

        psf_energies = np.empty([psf_decomposition.shape[0],1,1], dtype=np.float32)

        for i in range(psf_energies.shape[0]):
            psf_energies[i] = np.sqrt(np.sum(np.square(psf_decomposition[i,:,:])))

######################################################MAJOR LOOP######################################################

        major_loop_niter = 0
        max_coeff = 1

        model = np.zeros_like(self.dirty_data)

        std_current = 1
        std_last = 1
        std_ratio = 1

        min_scale = 0   # The current minimum scale of interest. If this ever equals or exceeds the scale_count
                        # value, it will also break the following loop.

        # In the case that edge_supression is desired, the following sets up a masking array.

        if edge_suppression:
            edge_corruption = 0
            suppression_array = np.zeros([scale_count,subregion,subregion],np.float32)
            for i in range(scale_count):
                edge_corruption += 2*2**i
                if edge_offset>edge_corruption:
                    suppression_array[i,edge_offset:-edge_offset, edge_offset:-edge_offset] = 1
                else:
                    suppression_array[i,edge_corruption:-edge_corruption, edge_corruption:-edge_corruption] = 1

        # The following is the major loop. Its exit conditions are reached if if the number of major loop iterations
        # exceeds a user defined value, the maximum wavelet coefficient is zero or the standard deviation of the
        # residual drops below a user specified accuracy threshold.

        while (((major_loop_niter<major_loop_miter) & (max_coeff>0)) & (std_ratio>accuracy)):

            # The first interior loop allows for the model to be re-estimated at a higher scale in the case of a poor
            # SNR. If, however, a better job cannot be done, the loop will terminate.

            while (min_scale<scale_count):

                # This is the IUWT decomposition of the dirty image subregion up to scale_count, followed by a
                # thresholding of the resulting wavelet coefficients based on the MAD estimator. This is a denoising
                # operation.

                if min_scale==0:
                    dirty_decomposition = iuwt.iuwt_decomposition(dirty_subregion, scale_count, 0, decom_mode, core_count)
                    dirty_decomposition_thresh = tools.threshold(dirty_decomposition, sigma_level=sigma_level)

                    # If edge_supression is desired, the following simply masks out the offending wavelet coefficients.

                    if edge_suppression:
                        dirty_decomposition_thresh *= suppression_array

                    # The following calculates and stores the normalised maximum at each scale.

                    normalised_scale_maxima = np.empty_like(psf_energies)

                    for i in range(dirty_decomposition_thresh.shape[0]):
                        normalised_scale_maxima[i] = np.max(dirty_decomposition_thresh[i,:,:])/psf_energies[i]

                # The following stores the index, scale and value of the global maximum coefficient.

                max_index = np.argmax(normalised_scale_maxima[min_scale:,:,:]) + min_scale
                max_scale = max_index + 1
                max_coeff = normalised_scale_maxima[max_index,0,0]

                # This is an escape condition for the loop. If the maximum coefficient is zero, then there is no
                # useful information left in the wavelets and MORESANE is complete.

                if max_coeff == 0:
                    logger.info("No significant wavelet coefficients detected.")
                    break

                logger.info("Minimum scale = {}".format(min_scale))
                logger.info("Maximum scale = {}".format(max_scale))

                # The following constitutes a major change to the original implementation - the aim is to establish
                # as soon as possible which scales are to be omitted on the current iteration. This attempts to find
                # a local maxima or empty scales below the maximum scale. If either is found, that scale all those
                # below it are ignored.

                scale_adjust = 0

                for i in range(max_index-1,-1,-1):
                    # if max_index > 1:
                    #     if (normalised_scale_maxima[i,0,0] > normalised_scale_maxima[i+1,0,0]):
                    #         scale_adjust = i + 1
                    #         logger.info("Scale {} contains a local maxima. Ignoring scales <= {}"
                    #                         .format(scale_adjust, scale_adjust))
                    #         break
                    if (normalised_scale_maxima[i,0,0] == 0):
                        scale_adjust = i + 1
                        logger.info("Scale {} is empty. Ignoring scales <= {}".format(scale_adjust, scale_adjust))
                        break

                # We choose to only consider scales up to the scale containing the maximum wavelet coefficient,
                # and ignore scales at or below the scale adjustment.

                thresh_slice = dirty_decomposition_thresh[scale_adjust:max_scale,:,:]

                # The following is a call to the externally defined source extraction function. It returns an array
                # populated with the wavelet coefficients of structures of interest in the image. This basically refers
                # to objects containing a maximum wavelet coefficient within some user-specified tolerance of the
                # maximum  at that scale.

                extracted_sources, extracted_sources_mask = \
                    tools.source_extraction(thresh_slice, tolerance, mode=extraction_mode, store_on_gpu=all_on_gpu)

                # The wavelet coefficients of the extracted sources are recomposed into a single image,
                # which should contain only the structures of interest.

                recomposed_sources = iuwt.iuwt_recomposition(extracted_sources, scale_adjust, decom_mode, core_count)

######################################################MINOR LOOP######################################################

                x = np.zeros_like(recomposed_sources)
                r = recomposed_sources.copy()
                p = recomposed_sources.copy()

                minor_loop_niter = 0

                snr_last = 0
                snr_current = 0

                # The following is the minor loop of the algorithm. In particular, we make use of the conjugate
                # gradient descent method to optimise our model. The variables have been named in order to appear
                # consistent with the algorithm.

                while (minor_loop_niter<minor_loop_miter):

                    Ap = conv.fft_convolve(p, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
                    Ap = iuwt.iuwt_decomposition(Ap, max_scale, scale_adjust, decom_mode, core_count,
                                                 store_on_gpu=all_on_gpu)
                    Ap = extracted_sources_mask*Ap
                    Ap = iuwt.iuwt_recomposition(Ap, scale_adjust, decom_mode, core_count)

                    alpha_denominator = np.dot(p.reshape(1,-1),Ap.reshape(-1,1))[0,0]
                    alpha_numerator = np.dot(r.reshape(1,-1),r.reshape(-1,1))[0,0]
                    alpha = alpha_numerator/alpha_denominator

                    xn = x + alpha*p

                    # The following enforces the positivity constraint which necessitates some recalculation.

                    if (np.min(xn)<0) & (enforce_positivity):

                        xn[xn<0] = 0
                        p = (xn-x)/alpha

                        Ap = conv.fft_convolve(p, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
                        Ap = iuwt.iuwt_decomposition(Ap, max_scale, scale_adjust, decom_mode, core_count,
                                                     store_on_gpu=all_on_gpu)
                        Ap = extracted_sources_mask*Ap
                        Ap = iuwt.iuwt_recomposition(Ap, scale_adjust, decom_mode, core_count)

                    rn = r - alpha*Ap

                    beta_numerator = np.dot(rn.reshape(1,-1), rn.reshape(-1,1))[0,0]
                    beta_denominator = np.dot(r.reshape(1,-1), r.reshape(-1,1))[0,0]
                    beta = beta_numerator/beta_denominator

                    p = rn + beta*p

                    model_sources = conv.fft_convolve(xn, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
                    model_sources = iuwt.iuwt_decomposition(model_sources, max_scale, scale_adjust, decom_mode,
                                                            core_count, store_on_gpu=all_on_gpu)
                    model_sources = extracted_sources_mask*model_sources

                    if all_on_gpu:
                        model_sources = model_sources.get()

                    # We compare our model to the sources extracted from the data.

                    snr_last = snr_current
                    snr_current = tools.snr_ratio(extracted_sources, model_sources)

                    minor_loop_niter += 1

                    logger.debug("SNR at iteration {0} = {1}".format(minor_loop_niter, snr_current))

                    # The following flow control determines whether or not the model is adequate and if a
                    # recalculation is required.

                    if (minor_loop_niter==1)&(snr_current>40):
                        logger.info("SNR too large on first iteration - false detection. "
                                    "Incrementing the minimum scale.")
                        min_scale += 1
                        break

                    if snr_current>40:
                        logger.info("Model has reached <1% error - exiting minor loop.")
                        x = xn
                        min_scale = 0
                        break

                    if (minor_loop_niter>2)&(snr_current<=snr_last):
                        if (snr_current>10.5):
                            logger.info("SNR has decreased - Model has reached ~{}% error - exiting minor loop."\
                                    .format(int(100/np.power(10,snr_current/20))))
                            min_scale = 0
                            break
                        else:
                            logger.info("SNR has decreased - SNR too small. Incrementing the minimum scale.")
                            min_scale += 1
                            break

                    r = rn
                    x = xn

                logger.info("{} minor loop iterations performed.".format(minor_loop_niter))

                if ((minor_loop_niter==minor_loop_miter)&(snr_current>10.5)):
                    logger.info("Maximum number of minor loop iterations exceeded. Model reached ~{}% error."\
                                    .format(int(100/np.power(10,snr_current/20))))
                    min_scale = 0
                    break

                if (min_scale==0):
                    break

###################################################END OF MINOR LOOP###################################################

            if min_scale==scale_count:
                logger.info("All scales are performing poorly - stopping.")
                break

            # The following handles the deconvolution step. The model convolved with the psf is subtracted from the
            # dirty image to give the residual.

            if max_coeff>0:

                model[subregion_slice] += loop_gain*x

                residual = self.dirty_data - conv.fft_convolve(model, psf_data_fft, conv_device, conv_mode)

                # The following assesses whether or not the residual has improved.

                std_last = std_current
                std_current = np.std(residual)
                std_ratio = (std_last-std_current)/std_last

                # If the most recent deconvolution step is poor, the following reverts the changes so that the
                # previous model and residual are preserved.

                if std_ratio<0:
                    logger.info("Residual has worsened - reverting changes.")
                    model[subregion_slice] -= loop_gain*x
                    residual = self.dirty_data - conv.fft_convolve(model, psf_data_fft, conv_device, conv_mode)

                # The current residual becomes the dirty image for the subsequent iteration.

                dirty_subregion = residual[subregion_slice]

                major_loop_niter += 1
                logger.info("{} major loop iterations performed.".format(major_loop_niter))

            # The following condition will only trigger if MORESANE did no work - this is an exit condition for the
            # by-scale approach.

            if (major_loop_niter==0):
                logger.info("Current MORESANE iteration did no work - finished.")
                self.complete = True
                break

        # If MORESANE did work at the current iteration, the following simply updates the values in the class
        # variables self.model and self.residual.

        if major_loop_niter>0:
            self.model += model
            self.residual = residual

    def moresane_by_scale(self, start_scale=1, stop_scale=20, subregion=None, sigma_level=4, loop_gain=0.1,
                          tolerance=0.75, accuracy=1e-6, major_loop_miter=100, minor_loop_miter=30, all_on_gpu=False,
                          decom_mode="ser", core_count=1, conv_device='cpu', conv_mode='linear', extraction_mode='cpu',
                          enforce_positivity=False, edge_suppression=False, edge_offset=0):
        """
        Extension of the MORESANE algorithm. This takes a scale-by-scale approach, attempting to remove all sources
        at the lower scales before moving onto the higher ones. At each step the algorithm may return to previous
        scales to remove the sources uncovered by the deconvolution.

        INPUTS:
        start_scale         (default=1)         The first scale which is to be considered.
        stop_scale          (default=20)        The maximum scale which is to be considered. Optional.
        subregion           (default=None):     Size, in pixels, of the central region to be analyzed and deconvolved.
        sigma_level         (default=4)         Number of sigma at which thresholding is to be performed.
        loop_gain           (default=0.1):      Loop gain for the deconvolution.
        tolerance           (default=0.75):     Tolerance level for object extraction. Significant objects contain
                                                wavelet coefficients greater than the tolerance multiplied by the
                                                maximum wavelet coefficient in the scale under consideration.
        accuracy            (default=1e-6):     Threshold on the standard deviation of the residual noise. Exit main
                                                loop when this threshold is reached.
        major_loop_miter    (default=100):      Maximum number of iterations allowed in the major loop. Exit
                                                condition.
        minor_loop_miter    (default=30):       Maximum number of iterations allowed in the minor loop. Serves as an
                                                exit condition when the SNR does not reach a maximum.
        all_on_gpu          (default=False):    Boolean specifier to toggle all gpu modes on.
        decom_mode          (default='ser'):    Specifier for decomposition mode - serial, multiprocessing, or gpu.
        core_count          (default=1):        In the event that multiprocessing, specifies the number of cores.
        conv_device         (default='cpu'):    Specifier for device to be used - cpu or gpu.
        conv_mode           (default='linear'): Specifier for convolution mode - linear or circular.
        extraction_mode     (default='cpu'):    Specifier for mode to be used - cpu or gpu.
        enforce_positivity  (default=False):    Boolean specifier for whether or not a model must be strictly positive.
        edge_suppression    (default=False):    Boolean specifier for whether or not the edges are to be suprressed.
        edge_offset         (default=0):        Numeric value for an additional user-specified number of edge pixels
                                                to be ignored. This is added to the minimum suppression.

        OUTPUTS:
        self.model          (no default):       Model extracted by the algorithm.
        self.residual       (no default):       Residual signal after deconvolution.
        """

        # The following preserves the dirty image as it will be changed on every iteration.

        dirty_data = self.dirty_data

        scale_count = start_scale


        while not (self.complete):

            logger.info("MORESANE at scale {}".format(scale_count))

            self.moresane(subregion=subregion, scale_count=scale_count, sigma_level=sigma_level, loop_gain=loop_gain,
                          tolerance=tolerance, accuracy=accuracy, major_loop_miter=major_loop_miter,
                          minor_loop_miter=minor_loop_miter, all_on_gpu=all_on_gpu, decom_mode=decom_mode,
                          core_count=core_count, conv_device=conv_device, conv_mode=conv_mode,
                          extraction_mode=extraction_mode, enforce_positivity=enforce_positivity,
                          edge_suppression=edge_suppression, edge_offset=edge_offset)

            self.dirty_data = self.residual

            scale_count += 1

            if (scale_count>(np.log2(self.dirty_data.shape[0]))-1):
                logger.info("Maximum scale reached - finished.")
                break

            if (scale_count>stop_scale):
                logger.info("Maximum scale reached - finished.")
                break

        # Restores the original dirty image.

        self.dirty_data = dirty_data
        self.complete = False

    def restore(self):
        """
        This method constructs the restoring beam and then adds the convolution to the residual.
        """
        clean_beam, beam_params = beam_fit.beam_fit(self.psf_data, self.psf_hdr)

        if np.all(np.array(self.psf_data_shape)==2*np.array(self.dirty_data_shape)):
            self.restored = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(conv.pad_array(self.model))*np.fft.rfft2(clean_beam)))
            self.restored = self.restored[self.dirty_data_shape[0]/2:-self.dirty_data_shape[0]/2,
                                          self.dirty_data_shape[1]/2:-self.dirty_data_shape[1]/2]
        else:
            self.restored = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(self.model)*np.fft.rfft2(clean_beam)))
        self.restored += self.residual
        self.restored = self.restored.astype(np.float32)

        self.img_hdr.update('BMAJ',beam_params[1])
        self.img_hdr.update('BMIN',beam_params[0])
        self.img_hdr.update('BPA',beam_params[2])

    def handle_input(self, input_hdr):
        """
        This method tries to ensure that the input data has the correct dimensions.

        INPUTS:
        input_hdr   (no default)    Header from which data shape is to be extracted.
        """

        input_slice = input_hdr['NAXIS']*[0]

        for i in range(input_hdr['NAXIS']):
            if input_hdr['CTYPE%d'%(i+1)].startswith("RA"):
                    input_slice[-1] = slice(None)
            if input_hdr['CTYPE%d'%(i+1)].startswith("DEC"):
                    input_slice[-2] = slice(None)

        return input_slice
    
    def save_fits(self, data, name):
        """
        This method simply saves the model components and the residual.

        INPUTS:
        data    (no default)    Data which is to be saved.
        name    (no default)    File name for new .fits file. Will overwrite.
        """
        data = data.reshape(1, 1, data.shape[0], data.shape[0])
        new_file = pyfits.PrimaryHDU(data,self.img_hdr)
        new_file.writeto(name, clobber=True)

    def make_logger(self, level="INFO"):
        """
        Convenience function which creates a logger for the module.

        INPUTS:
        level   (default="INFO"):   Minimum log level for logged/streamed messages.

        OUTPUTS:
        logger                      Logger for the function. NOTE: Must be bound to variable named logger.
        """
        level = getattr(logging, level.upper())

        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('PyMORESANE.log', mode='w')
        fh.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(''message)s', datefmt='[%m/%d/%Y] [%I:%M:%S]')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger
#--------------------------------------------------------------------------------
def fitsInfo(fits,return_data=False):
    """ Get some basic finfo from a FITS file """
    hdr = pyfits.open(fits)[0].header
    if return_data: 
        data = pyfits.open(fits)[0].data
    naxis = hdr['NAXIS']
    freq_ind = 0

    for i in range(1,naxis+1):
        if hdr['CTYPE%d'%i].upper().startswith('FREQ'): 
            freq_ind = i
    if freq_ind==0: 
        return (None,None),naxis,(None,None)

    nchan = hdr['NAXIS%d'%freq_ind]

    if return_data not in[False,None] : 
        return (freq_ind,nchan),naxis,(data,hdr)
    else: 
        return (freq_ind,nchan),naxis,(None,None)

def combine_fits(fitslist,outname='combined.fits',keep_old=False):
    """ Combine a list of fits files into a single cube """

    freqIndex = 0
    nchan = 0
    hdu = pyfits.open(fitslist[0])[0]
    hdr = hdu.header
    naxis = hdr['NAXIS']
    shape = list(hdu.data.shape)

    for key,val in hdr.iteritems():
        if key.startswith('CTYPE'):
            if val.upper().startswith('FREQ'): freqIndex = int(key[5:])
    if freqIndex ==0: 
        raise SystemError('At least one of the fits files has no\
frequency information in the header. Cannot combine fits images.')
    crval = hdr['CRVAL%d'%freqIndex]
    images = []

    for fits in fitslist:
        hdu = pyfits.open(fits)[0]
        hdr = hdu.header
        temp_crval = hdr['CRVAL%d'%freqIndex]
        nchan += hdr['NAXIS%d'%freqIndex]
        if temp_crval < crval : 
            crval = temp_crval
        images.append(hdu.data)

    ind = naxis - freqIndex # numpy array indexing differnt from FITS
    hdr['CRVAL%d'%freqIndex] = crval
    shape[ind] = nchan
    new_data = np.reshape(np.array(images),shape)
    pyfits.writeto(outname,new_data,hdr,clobber=True)
    if keep_old is False:
        for fits in fitslist:
            os.system('rm -rf %s'%fits)
#--------------------------------------------------------------------------------


if __name__ == "__main__":
    args = pparser.handle_parser()
   
    # Set up names for output images
    out_template = args.outputname or args.dirty
    model_image = args.model_image or out_template.replace('.fits','.model.fits')
    residual_image = args.residual_image or out_template.replace('.fits','.residual.fits')
    restored_image = args.restored_image or out_template.replace('.fits','.restored.fits')

    def run_moresane(data):
        if args.singlerun:
            data.moresane(args.subregion, args.scalecount, args.sigmalevel, 
                          args.loopgain, args.tolerance, args.accuracy,
                          args.majorloopmiter, args.minorloopmiter, 
                          args.allongpu, args.decommode, args.corecount,
                          args.convdevice, args.convmode, 
                          args.extractionmode, args.enforcepositivity,
                          args.edgesuppression, args.edgeoffset)
        else:
            data.moresane_by_scale(args.startscale, args.stopscale, 
                                   args.subregion, args.sigmalevel, args.loopgain,
                                   args.tolerance,args.accuracy, args.majorloopmiter, 
                                   args.minorloopmiter, args.allongpu, args.decommode,  
                                   args.corecount, args.convdevice, args.convmode,
                                   args.extractionmode, args.enforcepositivity, 
                                   args.edgesuppression, args.edgeoffset)

    start_time = time.time()

    dirty,psf = args.dirty,args.psf
    can_do_multi_chan = True

    (freq_axis_dirty,dirty_nchan),naxis_dirty,(dirty_data,dirty_hdr) = fitsInfo(dirty,return_data=True)
    if freq_axis_dirty == None: can_do_multi_chan = False

    (freq_axis_psf,psf_nchan),naxis_psf,(psf_data,psf_hdr) = fitsInfo(psf,return_data=True)
    if freq_axis_psf == None: can_do_multi_chan = False

    if np.logical_and(dirty_nchan==psf_nchan,psf_nchan>1) and can_do_multi_chan:
        print 'Dirty map and PSF map have %d frequency channels. Each channel will be deconvolved separately'
        # Get image data for each frequency slice, then send moresane
        ind_dirty = naxis_dirty - freq_axis_dirty
        ind_psf = naxis_psf - freq_axis_psf
        image_dirty = naxis_dirty*[0]
        image_psf = naxis_psf*[0]
        image_dirty[-2:] = [slice(None)]*2
        image_psf[-2:] = [slice(None)]*2
        # store moresane output images here, and combine them into a single cube later
        model_images,residual_images,restored_images = [],[],[] 

        for i in range(dirty_nchan):
            image_dirty[ind_dirty] = i
            image_psf[ind_psf] = i
            imageData = dirty_data[image_dirty]
            psfData = psf_data[image_psf]

            # Pass the full image hdr for each frequency slice. This header will be header for the combined fits
            # as well, with the frequency axis modified according,y
            data = FitsImage(args.dirty,imageData=imageData,imageHdr=dirty_hdr,psfData=psfData,psfHdr=psf_hdr)

            logger = data.make_logger(args.loglevel)
            logger.info("Parameters:\n" + str(args)[10:-1])
   
            run_moresane(data)      
    
            label = str(i).zfill(4)
            # add labels to images in multichannel mode
            model_im = out_template.replace('.fits','.model.%s.fits'%label) 
            residual_im = out_template.replace('.fits','.residual.%s.fits'%label) 
            restored_im = out_template.replace('.fits','.restored.%s.fits'%label) 

            data.save_fits(data.model, model_im)
            model_images.append(model_im)

            data.save_fits(data.residual, residual_im)
            residual_images.append(residual_im)

            data.restore()
            data.save_fits(data.restored, restored_im)
            restored_images.append(restored_im)

        # combine images into a single cube
        combine_fits(model_images,keep_old=False,outname=model_image)
        combine_fits(residual_images,keep_old=False,outname=residual_image)
        combine_fits(restored_images,keep_old=False,outname=restored_image)
        
        # make MFS and spi and/or spc maps
        if args.mfs:
            import specfit 
            if args.mfs_chanrange:
                start,end = map(int,args.mfs_chanrange.split(','))
            else:
                start,end = 0,len(model_images)-1
            specfit.get_spec(restored_image,spi=True,spc=args.spec_curv,start=start,end=end,sigma_level=args.spi_sigmalevel)

    # single channel case
    elif dirty_nchan==1:
        data = FitsImage(args.dirty, args.psf)

        logger = data.make_logger(args.loglevel)
        logger.info("Parameters:\n" + str(args)[10:-1])
        run_moresane(data)

        data.save_fits(data.model, model_image)
        data.save_fits(data.residual, residual_image)

        data.restore()
        data.save_fits(data.restored, restored_image)
    
    end_time = time.time()
    logger.info("Elapsed time was %s." % (time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))
  # test.moresane(scale_count = 9, major_loop_miter=100, minor_loop_miter=30, tolerance=0.8, \
  #                 conv_mode="linear", accuracy=1e-6, loop_gain=0.2, enforce_positivity=True, sigma_level=5,
  #                 decom_mode="gpu", extraction_mode="gpu", conv_device="gpu")
  # test.moresane_by_scale(major_loop_miter=100, minor_loop_miter=50, tolerance=0.75,
  #                 conv_mode="circular", accuracy=1e-6, loop_gain=0.3, enforce_positivity=True, sigma_level=4,
  #                 all_on_gpu=True, edge_suppression=True)
  # test.moresane_by_scale(subregion=512, major_loop_miter=100, minor_loop_miter=30, tolerance=0.7,
  #                 conv_mode="circular", accuracy=1e-6, loop_gain=0.2, enforce_positivity=True, sigma_level=4)
