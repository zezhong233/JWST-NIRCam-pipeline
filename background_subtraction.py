#  Iterative source removal and background subtraction
#

__author__ = "Henry C. Ferguson, STScI"
__version__ = "1.4.0"
__license__ = "BSD3"

# History
# 1.1.0 -- output the tier masks as a bitmask
#       -- Fixed bug where the number of masks was implicitly set to 4
# 1.1.1 -- Fixed bug where estimate of background rms in tier_mask was ignoring the mask
# 1.2.1 -- Appends the background-subtracted image as a separate extension rather than replacing the SCI
# 1.2.2 -- Makes replace_sci an option for the location of sky-subtracted image 
# 1.3.0 -- Optionally mask bits that are set in the DQ array, if it is there
# 1.3.1 -- Only pass tier_mask the mask for bad pixels and off-detector (not previous_mask)
# 1.4.0 -- Added clipped_ring_median to try to have the ring-median have less suppression in the outskirts of galaxies

import numpy as np
import gc 
from astropy.io import fits
from astropy import stats as astrostats
from os import path
from dataclasses import dataclass

# Imports for background estimation
from photutils.background import (
    Background2D,  # For estimating the background
    MedianBackground, BiweightLocationBackground, SExtractorBackground, # Not all are used
    BkgIDWInterpolator, BkgZoomInterpolator)  # For interpolating background
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint
from photutils.utils import ShepardIDWInterpolator as idw
from astropy.convolution import (
    convolve, convolve_fft, Box2DKernel, Tophat2DKernel,
    Ring2DKernel, Gaussian2DKernel)
from scipy.ndimage import median_filter
from astropy.wcs import WCS
import matplotlib.pyplot as plt
#import dill # Just for debugging

# If datamodels are used to allow selecting DQ flags by name
from jwst.datamodels import dqflags 

@dataclass
class SubtractBackground:
    tier_nsigma: list = (3., 3., 3., 3.)
    tier_npixels : list = (15, 15, 5, 2)
    tier_kernel_size: list = (25, 15, 5, 2)
    tier_dilate_size: list = (0, 0, 0, 3) 
    faint_tiers_for_evaluation: list = (3,4) # numbering starts at 1
    ring_radius_in:  float = 40
    ring_width: float = 3
    ring_clip_max_sigma: float = 5.
    ring_clip_box_size: int = 100  
    ring_clip_filter_size: int = 3
    bg_box_size: int = 5
    bg_filter_size: int = 3
    bg_exclude_percentile: int = 90
    bg_sigma: float = 3
    plot_smooth: int = 0
    suffix: str = "bkgsub"
    interpolator: str = 'zoom'
    replace_sci: bool = False
    dq_flags_to_mask: list = ('SATURATED',)

    def open_file(self,directory,fitsfile):
        with fits.open(path.join(directory,fitsfile)) as hdu:
            sci = hdu['SCI'].data
            try:
                err = hdu['ERR'].data
            except KeyError:
                # RMS map for HST
                err = hdu['RMS'].data
            self.has_dq = False
            for h in hdu:
                if 'EXTNAME' in h.header:
                    if h.header['EXTNAME'] == 'DQ':
                        self.has_dq = True
                        self.dq = h.data
                        print(f"{fitsfile} has a DQ array")
        return sci,err

    # Convenience routine for inspecting the background and mask
    def plot_mask(self, scene, bkgd, mask, zmin, zmax, smooth=0, slices=None):
        '''Make a three-panel plot of:
             * the mask for the whole image,
             * the scene times the mask
             * a zoomed-in region, with the mask shown as contours
        '''
        if slices:
            rows = slices[0]
            cols = slices[1]
            _mask = mask[rows,cols]
            _scene = scene[rows,cols]
            _bkgd = bkgd[rows,cols]
        else:
            _mask = mask
            _scene = scene
            _bkgd = bkgd
        plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.imshow(_mask, vmin=0, vmax=1, cmap=plt.cm.gray, origin='lower')
        plt.subplot(132)
        smooth = self.plot_smooth
        if smooth == 0:
            plt.imshow((_scene-_bkgd)*(1-_mask), vmin=zmin, vmax=zmax, origin='lower')
        else:
            smoothed = convolve((_scene-_bkgd)*(1-_mask), Gaussian2DKernel(smooth))
            plt.imshow(smoothed*(1-_mask), vmin=zmin/smooth, vmax=zmax/smooth,
                       origin='lower')
        plt.subplot(133)
        plt.imshow(_scene-_bkgd, vmin=zmin, vmax=zmax,origin='lower')
        plt.contour(_mask, colors='red', alpha=0.2)

    def replace_masked(self, sci, mask):
        sci_nan = np.choose(mask,(sci,np.nan)) #没有mask的地方用sci，mask掉的地方用nan代替，因为nan不会参与到biweight_location的计算中
        robust_mean_background = astrostats.biweight_location(sci_nan,c=6.,ignore_nan=True)
        sci_filled = np.choose(mask,(sci,robust_mean_background))# mask掉的地方用robust_mean_bkg代替，没有被mask掉的地方用sci
        return sci_filled

    def off_detector(self, sci, err):
        return np.isnan(err) # True if OFF detector, False if on detector

    def mask_by_dq(self):
        self.dqmask = np.zeros(len(self.dq),bool)
        for flag_name in self.dq_flags_to_mask:
            flagbit = dqflags.pixel[flag_name]
            self.dqmask = self.dqmask | (np.bitwise_and(self.dq,flagbit) != 0)

    def clipped_ring_median_filter(self, sci, mask):
        print("Start ring median filter.")
        # First make a smooth background (clip_box_size should be big)
        bkg = Background2D(sci,
              box_size = self.ring_clip_box_size,
              sigma_clip = astrostats.SigmaClip(sigma=self.bg_sigma),
              filter_size = self.ring_clip_filter_size,
              bkg_estimator = BiweightLocationBackground(),
              exclude_percentile = 90,
              mask = mask,
              interpolator = BkgZoomInterpolator())
        print("bkg estimation finished.")
        # Estimate the rms after subtracting this
        background_rms = astrostats.biweight_scale((sci-bkg.background)[~mask]) 
        # Apply a floating ceiling to the original image
        ceiling = self.ring_clip_max_sigma * background_rms + bkg.background
        # Pixels above the ceiling are masked before doing the ring-median filtering
        ceiling_mask = sci > ceiling 
        print(f"Ring median filtering with radius, width = ",end='')
        print(f"{self.ring_radius_in}, {self.ring_width}")
        sci_filled = self.replace_masked(sci, mask | ceiling_mask) # | 代表有一个是True就是True，这句话的意思是把mask和ceiling_mask mask 掉的地方用平均值代替
        ring = Ring2DKernel(self.ring_radius_in, self.ring_width)
        filtered = median_filter(sci_filled, footprint=ring.array) 
        print("ring median filter finished.")

        del ceiling_mask
        del sci_filled 
        gc.collect()

        return sci-filtered

    def tier_mask(self, img, mask, tiernum = 0):
        background_rms = astrostats.biweight_scale(img[~mask]) # Already has been ring-median subtracted
        # Replace the masked pixels by the robust background level so the convolution doesn't smear them
        background_level = astrostats.biweight_location(img[~mask]) # Already has been ring-median subtracted
        replaced_img = np.choose(mask,(img,background_level))
        print("replaces_img generated")
        convolved_difference = convolve_fft(replaced_img,Gaussian2DKernel(self.tier_kernel_size[tiernum]),allow_huge=True)
        print("FFT convolution finished.")

        # First detect the sources, then make masks from the SegmentationImage
        seg_detect = detect_sources(convolved_difference, 
                    threshold=self.tier_nsigma[tiernum] * background_rms, #in do_background_subtraction function, it will subtract the ring median filtered background, and the bkg_level is approxmately 0.
                    npixels=self.tier_npixels[tiernum], 
                     mask=mask)
        print("source_detection finished.")
        if self.tier_dilate_size[tiernum] == 0:
            if seg_detect is None:
                return mask
            else:
                mask = seg_detect.make_source_mask()
        else:
            footprint = circular_footprint(radius=self.tier_dilate_size[tiernum])
            if seg_detect is None:
                print("这一轮啥也没有。")
                return mask
            else:
                mask = seg_detect.make_source_mask(footprint = footprint)
        print(f"Tier #{tiernum}:")

        return mask

    def mask_sources(self, img, mask): 
        ''' Iteratively mask sources 
            Wtarting_bit lets you add bits for these masks to an existing bitmask
        '''
        print(f"ring-filtered background median: {np.nanmedian(img)}")

        for tiernum in range(len(self.tier_nsigma)):
            current_mask = self.tier_mask(img, mask, tiernum=tiernum)
            mask = np.logical_or(current_mask, mask)
            
        return mask
    
    def estimate_background(self, img, mask):
        bkg = Background2D(img, 
                    box_size = self.bg_box_size,
                    sigma_clip = astrostats.SigmaClip(sigma=self.bg_sigma),
                    filter_size = self.bg_filter_size,
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = self.bg_exclude_percentile,
                    mask = mask,
                    interpolator = BkgZoomInterpolator())
        return bkg
    
    def estimate_background_IDW(self, img, mask):
        bkg = Background2D(img, 
                    box_size = self.bg_box_size,
                    sigma_clip = astrostats.SigmaClip(sigma=self.bg_sigma),
                    filter_size = self.bg_filter_size,
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = self.bg_exclude_percentile,
                    mask = mask,
                    interpolator = BkgIDWInterpolator())
        return bkg
    # Customize the parameters for the different steps here 
    def do_background_subtraction(self, datadir, fitsfile,  outputpath):
        # Background subtract all the bands
        print(fitsfile)
        sci, err = self.open_file(datadir,fitsfile)
        print("读取出了sci和err")
        # Set up a bitmask
        # bitmask = np.zeros(sci.shape,np.uint32) # Enough for 32 tiers
        mask = np.zeros(sci.shape, bool)
        print("设置空mask")

        # First level is for masking pixels off the detector
        off_detector_mask = self.off_detector(sci,err)#把不在探测器上的像素mask掉
        #bitmask = np.bitwise_or(bitmask,np.left_shift(off_detector_mask,0))
        print("off detector 的像素被mask掉")
        # Mask by DQ bits if desired and DQ file exists
        if self.has_dq:
            self.mask_by_dq()
            mask = off_detector_mask | self.dqmask
        else:
            mask = off_detector_mask 


        del off_detector_mask
        gc.collect()
        # bitmask = np.bitwise_or(bitmask,np.left_shift(mask,0))    

        # Ring-median filter 
        filtered = self.clipped_ring_median_filter(sci, mask) #filter 是去掉背景之后的data

        print(f"Finished the ring_median_filter for {fitsfile}")

        # Mask sources iteratively in tiers
        print("开始tier mask")
        mask = self.mask_sources(filtered, mask)
        print("结束tier mask")

        del filtered
        gc.collect()

        # Estimate the background using just unmasked regions
        if self.interpolator == 'IDW':
            bkg = self.estimate_background_IDW(sci, mask)
        else:
            bkg = self.estimate_background(sci, mask)
        bkgd = bkg.background

        # Subtract the background
        bkgd_subtracted = sci-bkgd

        # Write out the results
        prefix = fitsfile[:fitsfile.rfind('_')] #   找到最后一个_之前的所有字符
        outfile = f"{prefix}_{self.suffix}.fits"
        outpath = path.join(outputpath,outfile)
        hdu = fits.open(path.join(datadir,fitsfile))
        wcs = WCS(hdu['SCI'].header) # Attach WCS to it
        # Replace or append the background-subtracted image
        # Replace
        if self.replace_sci:
            hdu['SCI'].data = bkgd_subtracted
        # Append 
        else: 
            newhdu = fits.ImageHDU(bkgd_subtracted,header=wcs.to_header(),name='BKGSUB')
            hdu.append(newhdu)   
        # Append an extension with the bitmask from the tiers of source rejection
        # newhdu = fits.ImageHDU(bitmask,header=wcs.to_header(),name='TIERMASK')
        mask = mask.astype(np.uint8)
        newhdu = fits.ImageHDU(mask, header = wcs.to_header(), name = "MASK")
        hdu.append(newhdu)
        # Write out the new FITS file
        hdu.writeto(outpath,overwrite=True)
        print(f"Writing out {outpath}")
        print("")
 