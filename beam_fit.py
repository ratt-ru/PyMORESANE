import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import pylab as pb

def beam_fit(psf, psf_header):
    """
    The following contructs a restoring beam from the psf. This is accoplished by fitting an elliptical Gaussian to the
    central lobe of the PSF.

    INPUTS:
    psf         (no default):   Array containing the psf for the image in question.
    psf_header  (no default):   Header of the psf.
    """

    if psf.shape>512:
        psf_slice = tuple([slice(psf.shape[0]/2-256, psf.shape[0]/2+256),slice(psf.shape[1]/2-256, psf.shape[1]/2+256)])
    else:
        psf_slice = tuple([slice(0, psf.shape[0]),slice(0, psf.shape[1])])

    psf_centre = psf[psf_slice]/np.max(psf[psf_slice])

    max_location = np.unravel_index(np.argmax(psf_centre), psf_centre.shape)

    threshold_psf = np.where(psf_centre>0.1, psf_centre, 0)

    labelled_psf, labels = ndimage.label(threshold_psf)

    extracted_primary_beam = np.where(labelled_psf==labelled_psf[max_location], psf_centre, 0)

    # Following creates row and column values of interest for the central PSF lobe and then selects those values
    # from the PSF using np.where. Additionally, the inputs for the fitting are created by reshaping the x,y,
    # and z data into columns.

    x = np.arange(-max_location[1],-max_location[1]+psf_centre.shape[1],1)
    y = np.arange(-max_location[0],-max_location[0]+psf_centre.shape[0],1)
    z = extracted_primary_beam

    gridx, gridy = np.meshgrid(x,-y)

    xyz = np.column_stack((gridx.reshape(-1,1),gridy.reshape(-1,1),z.reshape(-1,1,order="C")))

    # Elliptical gaussian which can be fit to the central lobe of the PSF. xy must be an Nx2 array consisting of
    # pairs of row and column values for the region of interest.

    def ellipgauss(xy,A,xsigma,ysigma,theta):
        return A*np.exp(-1*(((xy[:,0]*np.cos(theta)-xy[:,1]*np.sin(theta))**2)/(2*(xsigma**2)) +
                            ((xy[:,0]*np.sin(theta)+xy[:,1]*np.cos(theta))**2)/(2*(ysigma**2))))

    # This command from scipy performs the fitting of the 2D gaussian, and returns the optimal coefficients in opt.

    opt = curve_fit(ellipgauss, xyz[:,0:2],xyz[:,2],(1,1,1,0))[0]

    # Following create the data for the new images. The cleanbeam has to be reshaped to reclaim it in 2D.

    clean_beam = np.zeros_like(psf)
    clean_beam[psf_slice] = ellipgauss(xyz[:,0:2],opt[0],opt[1],opt[2],opt[3]).reshape(psf_centre.shape,order="C")

    # Experimental - forces the beam to be normalised. This should be redundant, but helps when the PSF is bad.

    clean_beam = clean_beam/np.max(clean_beam)

    bmaj = 2*np.sqrt(2*np.log(2))*max(opt[1],opt[2])*psf_header['CDELT1']
    bmin = 2*np.sqrt(2*np.log(2))*min(opt[1],opt[2])*psf_header['CDELT2']
    bpa = np.degrees(opt[3])%360 - 90

    beam_params = [abs(bmaj), abs(bmin), bpa]

    return clean_beam, beam_params
