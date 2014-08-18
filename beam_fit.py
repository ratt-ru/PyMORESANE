import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import pylab as pb

def beam_fit(psf,psfheader):
    """
    The following contructs a restoring beam from the psf. This is accoplished by fitting a Gaussian to the central
    lobe. Accpets the following paramters:

    psf     (no default):   Array containing the psf for the image in question.
    """
    psf_max_location = np.unravel_index(np.argmax(psf), psf.shape)

    threshold_psf = np.where(psf>0.1,psf,0)

    labelled_psf, labels = ndimage.label(threshold_psf)

    labelled_primary_beam = np.where(labelled_psf==labelled_psf[psf_max_location],1,0)

    primary_beam_slice = ndimage.find_objects(labelled_primary_beam)[0]
    extracted_primary_beam = threshold_psf[primary_beam_slice]

    extracted_primary_beam_shape = extracted_primary_beam.shape
    extracted_primary_beam_max_location = np.unravel_index(np.argmax(extracted_primary_beam),extracted_primary_beam_shape)

    # Following creates row and column values of interest for the central PSF lobe and then selects those values
    # from the PSF using np.where. Additionally, the inputs for the fitting are created by reshaping the x,y,
    # and z data into columns.

    x = np.arange(-extracted_primary_beam_max_location[1],-extracted_primary_beam_max_location[
        1]+extracted_primary_beam_shape[1],1)
    y = np.arange(-extracted_primary_beam_max_location[0],-extracted_primary_beam_max_location[
        0]+extracted_primary_beam_shape[0],1)
    z = extracted_primary_beam

    gridx, gridy = np.meshgrid(x,-y)

    xyz = np.column_stack((gridx.reshape(-1,1),gridy.reshape(-1,1),z.reshape(-1,1,order="C")))

    # Elliptical gaussian which can be fit to the central lobe of the PSF. xy must be an Nx2 array consisting of
    # pairs of row and column values for the region of interest. This is obtained from kk above.

    def ellipgauss(xy,A,xsigma,ysigma,theta):
        return A*np.exp(-1*(((xy[:,1]*np.cos(theta)-xy[:,0]*np.sin(theta))**2)/(2*(xsigma**2))+(xy[:,1]*np.sin(theta)+(
            xy[:,0]*np.cos(theta))**2)/(2*(ysigma**2))))

    # This command from scipy performs the fitting of the 2D gaussian, and returns the optimal coefficients in opt.

    opt = curve_fit(ellipgauss, xyz[:,0:2],xyz[:,2],(1,1,1,0))[0]

    # Following create the data for the new images. The cleanbeam has to be reshaped to reclaim it in 2D.

    cleanbeam = np.zeros_like(psf)
    cleanbeam[primary_beam_slice] = ellipgauss(xyz[:,0:2],opt[0],opt[1],opt[2],opt[3]).reshape(gridx.shape,order="C")

    rotationangle = np.degrees(opt[3])
    FWHMmajor = 2*np.sqrt(2*np.log(2))*max(opt[1],opt[2])*psfheader[0].header['CDELT1']
    FWHMminor = 2*np.sqrt(2*np.log(2))*min(opt[1],opt[2])*psfheader[0].header['CDELT2']

    beamparams = [rotationangle,abs(FWHMmajor),abs(FWHMminor)]

    return labelled_primary_beam*psf, beamparams
