from scipy import ndimage
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from photutils.segmentation import detect_threshold, detect_sources
from astropy.convolution import Gaussian2DKernel, convolve

import numpy as np

 
def make_source_mask(data, nsigma, npixels, mask=None, filter_fwhm=None,
                     filter_size=3, kernel=None, sigclip_sigma=3.0,
                     sigclip_iters=5, dilate_size=11):
        

    sigma_clip = SigmaClip(sigma=sigclip_sigma, maxiters=sigclip_iters)
    threshold = detect_threshold(data, nsigma, background=None, error=None,
                                 mask=mask, sigma_clip=sigma_clip)
 
    if kernel is None and filter_fwhm is not None:
        kernel_sigma = filter_fwhm * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(kernel_sigma, x_size=filter_size,
                                  y_size=filter_size)
    if kernel is not None:
        kernel.normalize()

    print("data' shape is:", data.shape)
    if kernel is None:
        con_data = data
    else:
        con_data = convolve(data, kernel=kernel, mask = mask, normalize_kernel = True)
   
    segm = detect_sources(con_data, threshold, npixels)
    if segm is None:
        return np.zeros(data.shape, dtype=bool)

    selem = np.ones((dilate_size, dilate_size))
    return ndimage.binary_dilation(segm.data.astype(bool), selem)