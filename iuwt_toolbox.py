import numpy as np
from scipy import ndimage

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

def source_extraction(in1, max_coeff, tolerance):
    """
    This function replicates the functionality of src_ect.m and object_extraction.m. It constructs a matrix of
    connected components using the functionality of ndimage. Each component is examined at each scale and components
    with local maxima greater than the tolerance multiplied by the absolute maximum are flagged as objects. These
    objects are "extracted" - and returned to the parent function for further manipulation. Further investigation is
    required to decide whether finding connections in 3D is better. The source extraction kernel is recursive in
    nature, but it may be possible to further optimise the procedure using 3D connectedness. Accepts the following
    paramters:

    IUWT        (no default):   The wavelet decomposition from which the objects sources should be extracted.
    maxscale    (no default):   The maximum scale of of the decomposition.
    IUWTsize    (no default):   The size of IUWT.
    maxcoeff    (no default):   The value of the maximum wavelet coefficient.
    IUWT3sigma  (no default):   The decomposition at maxscale using 3 sigma thresholding.
    tolerance   (no default):   User defined tolerance which determines when objects are of significance.
    """

    # TODO: Consider finding connectedness in 3D - may make code substantially faster.

    # Initialisation of the variables in which the connected components are stored, as well as the calculated local
    # maxima and their locations. The absolute maximum for the scale is stored in pagemax.

    objects = np.empty_like(in1, dtype=int)               # Variable to store the objects found by ndimage.
    object_count = np.empty([in1.shape[0],1], dtype=int)  # Variable to store the number of objects in each scale.
    scale_maxima = np.empty([in1.shape[0],1])             # Variable of convenience storing the maximum of each scale.

    # Objects is a multi-page numpy array containing the ndimage label array for a given scale.
    # objectparams contains: the scale, the position of the maximum coefficient, its value, its label and the label
    # of the object in the next scale.

    for i in range(in1.shape[0]):

        # Finds the objects at each scale as well as the maximum wavelet coefficient for the scale.

        scale_maxima[i] = np.max(in1[i,:,:])

        objects[i,:,:], object_count[i] = ndimage.label(in1[i,:,:], structure=[[1,1,1],[1,1,1],[1,1,1]])

    object_params = np.empty([np.sum(object_count), 5], dtype=np.int)
    object_maxima = np.empty([np.sum(object_count), 1], dtype=np.float32)
    scale_index = np.cumsum(object_count)

    lower_bound = 0
    upper_bound = 0

    # This loop populates object_params with the important values pertaining to each object. In order, these are:
    # scale, row of max, col of max, max, label at scale, label at scale+1.

    for i in range(in1.shape[0]):

        upper_bound = scale_index[i]

        object_params[lower_bound:upper_bound,0] = i

        object_params[lower_bound:upper_bound,1:3] = \
            ndimage.maximum_position(in1[i,:,:], objects[i,:,:], index=np.arange(1, object_count[i] + 1))

        object_params[lower_bound:upper_bound,3] = \
            objects[object_params[lower_bound:upper_bound,0],
                    object_params[lower_bound:upper_bound,1],
                    object_params[lower_bound:upper_bound,2]]

        object_maxima[lower_bound:upper_bound,0] = \
            in1[i, object_params[lower_bound:upper_bound,1],
                   object_params[lower_bound:upper_bound,2]]

        if i!=(in1.shape[0]-1):
            object_params[lower_bound:upper_bound,4] = \
                objects[object_params[lower_bound:upper_bound,0]+1,
                        object_params[lower_bound:upper_bound,1],
                        object_params[lower_bound:upper_bound,2]]

        lower_bound = scale_index[i]

    # Initialises variables to store the source count as we all an empty list into which the found components can be
    # stacked.

    source_number = 1
    found_sources = []

    # Determines whether or not a component is significant - if it is, its values at all scales are found and checked
    # for significance individually. Only sources identified in maxscale can be classed as significant.

    for i in range(-object_count[-1],0):

        # The first step asses the significance of objects at maxscale. Significant objects have their values
        # extracted and stored in foundobjects which contains the source number, the scales at which it appears and
        # its labels at those scales.

        if object_maxima[i,0]>(max_coeff*tolerance):
            object_label = object_params[i,3]
            scale_number = in1.shape[0]

            found_sources.append(source_number)
            found_sources.append(scale_number)
            found_sources.append(object_label)

            # This is a call to the recursive source extraction kernel which essentially creates a tree of
            # significance for each object found at maxscale - it cascades back through lower scales finding
            # significant components of the object at maxscale.

            found_sources = source_extraction_kernel(found_sources, object_params, object_maxima, object_count,
                                                     source_number, scale_number-1, scale_maxima, tolerance)

            source_number += 1

    found_sources = np.asarray(found_sources).reshape([-1,3])

    # The following generates the mask for objects not at the maxscale. It simply places ones in the mask for each
    # label in found objects, at the appropriate scale.

    extracted_sources_mask = np.zeros_like(in1)

    for i in found_sources:
        extracted_sources_mask[i[1]-1,:,:] += np.where(objects[i[1]-1,:,:]==i[2],1,0)

    # The following fills the array with the values belonging to the extracted objects.

    extracted_sources = in1*extracted_sources_mask

    print found_sources

    return extracted_sources, scale_maxima

def source_extraction_kernel(found_sources, object_params, object_maxima, object_count, source_number,
                             scale_number, scale_maxima, tolerance):
    """
    This is the source extraction kernel, a recursive function which finds the components of objects significant at
    maxscale at lower scales. Accpets the following parameters:

    foundobjects    (no default):   List containing the currently found objects.
    objectparams    (no default):   The parameters of all the found objects.
    objectcount     (no default):   The number of objects found in each scale.
    sourcenumber    (no default):   The number of the source.
    scale           (no default):   The scale which is currently being analysed.
    pagemax         (no default):   The maximum wavelet coefficients for each scale.
    tolerance       (no default):   User defined tolerance which determines when objects are of significance.
    """

    # Recursion stops once the scale value is zero.

    if scale_number==0:
        return

    # The end value of the list corresponds to the label of interest in scale+1.

    object_label = found_sources[-1]

    # The following loops over the values for objects at the current scale and appends to the list if it finds the
    # label. If the coefficient at the new scale is also significant, this function is called recursively to find
    # what the object at the new scale is connected to.

    for i in range(np.sum(object_count[:scale_number-1]), np.sum(object_count[:scale_number])):

        if object_params[i,4]==object_label:

            found_sources.append(source_number)
            found_sources.append(scale_number)
            found_sources.append(object_params[i,3])

            if object_maxima[i,0]>(scale_maxima[scale_number-1]*tolerance):

                source_extraction_kernel(found_sources, object_params, object_maxima, object_count,
                                         source_number, scale_number-1, scale_maxima, tolerance)

    return found_sources

def source_extraction_rev(in1, tolerance):

    objects = np.empty_like(in1, dtype=int)
    object_count = np.empty([in1.shape[0],1], dtype=int)
    scale_maxima = np.empty([in1.shape[0],1])

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
    extraction1 = source_extraction_rev(thresh_decom, 0.1)
    print time.time() - t

    for i in range(extraction1.shape[0]):
        pb.imshow(extraction1[i,:,:])
        pb.show()

    t = time.time()
    extraction2, maxima = source_extraction(thresh_decom, np.max(thresh_decom), 0.1)
    print time.time() - t

    for i in range(extraction2.shape[0]):
        pb.imshow(extraction2[i,:,:])
        pb.show()

    print np.allclose(extraction1,extraction2,1e-6)
