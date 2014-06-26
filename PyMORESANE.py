import pyfits
import numpy as np
import math
# import iuwt
import argparse
from scipy.signal import fftconvolve
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

    def moresane(self, subregion=None, scale_count=None, loop_gain=0.1, tolerance=0.7, accuracy=1e-6,
                 loop_max_iter=100, decon_max_iter=30, func_mode='ser', conv_mode='linear'):
        """
        Primary method for wavelet analysis and subsequent deconvolution.

        INPUTS:
        subregion       (default=None):     Size, in pixels, of the central region to be analyzed and deconvolved.
        scale_count     (default=None):     Maximum scale to be considered - maximum scale considered during
                                            initialisation.
        loop_gain       (default=0.1):      Loop gain for the deconvolution.
        tolerance       (default=0.7):      Tolerance level for object extraction. Significant objects contain
                                            wavelet coefficients greater than the tolerance multiplied by the
                                            maximum wavelet coefficient in the scale under consideration.
        accuracy        (default=1e-6):     Threshold on the standard deviation of the residual noise. Exit main
                                            loop when this threshold is reached.
        loop_max_iter   (default=100):      Maximum number of iterations allowed in the major loop. Exit condition.
        decon_max_iter  (default=30):       Maximum number of iterations allowed in the minor loop. Serves as an
                                            exit condition when the SNR is does not reach a maximum.
        func_mode       (default='ser'):    Specifier for mode to be used - serial, multiprocessing or gpu.
        conv_mode       (default='linear'): Specifier for convolution mode - linear or circular.

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

        # TODO: Reimplement the contents of StationaryWaveletDecomposition not in iuwt in iuwt_toolbox.

        pass

if __name__ == "__main__":
    test = FitsImage("3C147.fits","3C147_PSF.fits")

    start_time = time.time()
    test.moresane()
    end_time = time.time()
    print("Elapsed time was %g seconds" % (end_time - start_time))
