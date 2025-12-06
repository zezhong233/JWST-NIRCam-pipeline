from photutils.background import Background2D, BiweightLocationBackground, BkgIDWInterpolator, BkgZoomInterpolator
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint
from astropy.convolution import convolve, convolve_fft, Ring2DKernel, Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
import astropy.stats as astrostats
import numpy as np
from scipy.ndimage import median_filter
import os
from datetime import datetime
from dataclasses import dataclass
@dataclass
class BackgroundSubtraction():
    factor = 1.5
    ring_radius_in = 80
    ring_width = 4
    ring_clip_box_size = 100
    tier_kernel_size:list = (25, 15, 5, 2)
    bg_filter_size = 5
    bg_box_size = 10
    tier_dilate_size = [33, 25, 21, 19]
    tier_nsigma = [1.5, 1.5, 1.5, 1.5]
    tier_npixels = [15, 10, 4, 2]

    def open_file(self, data_dir, fits_file):
        print(fits_file)
        with fits.open(os.path.join(data_dir, fits_file)) as hdul:
            sci = hdul["SCI"].data
            err = hdul["ERR"].data
        return sci, err
        
    def replace_mask(self,sci, mask):
        sci_nan = np.choose(mask, (sci, np.nan))
        robust_median = astrostats.biweight_location(sci_nan, ignore_nan = True)
        sci_robust = np.choose(mask, (sci, robust_median))
        return sci_robust
    
    def off_detector(self, err):
        return np.isnan(err)

    def mask_by_dp(self):
        pass

    def clipped_ring_median_filter(self, sci, mask):
        bkg = Background2D(sci, 
                           box_size=100, sigma_clip=astrostats.SigmaClip(sigma = 3),
                            filter_size = 3,  bkg_estimator =  BiweightLocationBackground(),
                            exclude_percentile=90,mask = mask,interpolator =  BkgZoomInterpolator())  
        background_rms = astrostats.biweight_scale((sci-bkg.background)[~mask])
        ceiling = 5*background_rms + bkg.background

        ceiling_mask = sci > ceiling
        sci_filled = self.replace_mask(sci, (mask | ceiling_mask))
        ring = Ring2DKernel(self.ring_radius_in, self.ring_width)
        filtered = median_filter(sci_filled, footprint = ring.array)

        return sci - filtered

    def tier_mask(self, sci, mask, tier_num = 0):

        bkg_rms = astrostats.biweight_scale(sci[~mask])
        bkg_level = astrostats.biweight_location(sci[~mask])
        replaced_sci = np.choose(mask, (sci, bkg_level))
        convolved_sci = convolve_fft(replaced_sci, Gaussian2DKernel(self.tier_kernel_size[tier_num] * self.factor), allow_huge=True)

        seg_detect = detect_sources(convolved_sci, threshold = self.tier_nsigma[tier_num] * bkg_rms, npixels = int(self.tier_npixels[tier_num] * self.factor)+1,
                                    mask = mask)
        footprint = circular_footprint(radius = int(self.tier_dilate_size[tier_num] * self.factor))
        mask = seg_detect.make_source_mask(footprint = footprint)

        print(f"Tier #{tier_num}:")
        print(f"  kernel_size = {self.factor * self.tier_kernel_size[tier_num]}")
        print(f"  tier_nsigma = {self.tier_nsigma[tier_num]}")
        print(f"  tier_npixels = {self.factor * self.tier_npixels[tier_num]}")
        print(f"  tier_dilate_size = {self.factor * self.tier_dilate_size[tier_num]}")
        print(f"  median of ring-median-filtered image = {np.nanmedian(sci)}")
        print(f"  biweight rms of ring-median-filtered image  = {bkg_rms}")

        return mask

    def mask_sources(self,sci, mask, bitmask, start_bit = 1):

        for tier_num in range(len(self.tier_dilate_size)):
            print(f"mask tier:{tier_num}")
            mask_cur = self.tier_mask(sci, mask, tier_num = tier_num)
            bitmask = np.bitwise_or(bitmask,np.left_shift(mask_cur,tier_num+start_bit))
            mask = mask | mask_cur
        return mask, bitmask
    
    def estimate_background_IDW(self, img, mask):
        bkg = Background2D(img, 
                    box_size = self.bg_box_size,
                    sigma_clip = astrostats.SigmaClip(sigma=3),
                    filter_size = self.bg_filter_size,
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = 90,
                    mask = mask,
                    interpolator = BkgIDWInterpolator())
        return bkg

    def evaluate_bias(self, bkgd, err, mask):

        on_detector = np.logical_not(np.isnan(err)) # True if on detector, False if not
    
        mean_masked = bkgd[mask & on_detector].mean()
        std_masked = bkgd[mask & on_detector].std()
        stderr_masked = mean_masked/(np.sqrt(len(bkgd[mask]))*std_masked)
    
        mean_unmasked = bkgd[~mask & on_detector].mean()
        std_unmasked = bkgd[~mask & on_detector].std()
        stderr_unmasked = mean_unmasked/(np.sqrt(len(bkgd[~mask]))*std_unmasked)
        
        diff = mean_masked - mean_unmasked
        significance = diff/np.sqrt(stderr_masked**2 + stderr_unmasked**2)
        
        print(f"Mean under masked pixels   = {mean_masked:.8f} +- {stderr_masked:.8f}")
        print(f"Mean under unmasked pixels = "
              f"{mean_unmasked:.8f} +- {stderr_unmasked:.8f}")
        print(f"Difference = {diff:.8f} at {significance:.8f} sigma significance")
    
    def do_background_subtraction(self, data_dir, fits_file, output_path, evaluate = False, merged_mask_path = None):

        start_time = datetime.now()

        sci, err = self.open_file(data_dir=data_dir, fits_file = fits_file)

        if merged_mask_path :
            with fits.open(merged_mask_path) as hdul:
                mask = hdul[0].data

        else:
            off_detmask = self.off_detector(err)

            bitmask = np.zeros(sci.shape,np.uint32)
            bitmask = np.bitwise_or(bitmask,np.left_shift(off_detmask,0))

            sci_filtered = self.clipped_ring_median_filter(sci, off_detmask)
            print("ring median filter finished.")
            mask, bitmask = self.mask_sources(sci_filtered, off_detmask, bitmask)
            print("tier mask finished.")

        bkg = self.estimate_background_IDW(sci, mask)
        bkgd = bkg.background
        bkg_rms = bkg.background_rms
        bkg_sub = sci - bkgd 
        print("bkgsub finished.")
        # Evaluate
        if evaluate:
            print("Bias under bright sources:")
            self.evaluate_bias(bkgd,err,mask) # Under all the sources
            print("\nBias under fainter sources")
            faintmask = np.zeros(sci.shape,bool)
            for t in [3,4]:
                faintmask = faintmask | (np.bitwise_and(bitmask,2**t) != 0)
            self.evaluate_bias(bkgd,err,faintmask) # Just under the fainter sources
            print("evaluation finished.")
        ## readout

        hdul_old = fits.open(os.path.join(data_dir, fits_file))
        hdul = fits.HDUList([fits.PrimaryHDU()])
        wcs = WCS(hdul_old["SCI"].header)
        if not merged_mask_path:
            bkgsub_hdu = fits.ImageHDU(bkg_sub.astype(np.float32), header = hdul_old["SCI"].header, name = "BKGSUB")
            bitmask_hdu = fits.ImageHDU(bitmask.astype(np.int32), header = hdul_old["SCI"].header, name = "TIERMASK")
            bkgd_hdu = fits.ImageHDU(bkgd.astype(np.float32), header = hdul_old["SCI"].header, name = "BACKGROUND")
            bkgd_rms_hdu = fits.ImageHDU(bkg_rms.astype(np.float32), header = hdul_old["SCI"].header, name = "BACKGROUND_RMS")

            hdul.append(bkgsub_hdu)
            hdul.append(bkgd_hdu)
            hdul.append(bitmask_hdu)
            hdul.append(bkgd_rms_hdu)

            hdul.writeto(output_path, overwrite = True)
            
        else:
            bkgsub_hdu = fits.ImageHDU(bkg_sub.astype(np.float32), header = hdul_old["SCI"].header, name = "M_BKGSUB")
            merge_mask_hdu = fits.ImageHDU(mask.astype(np.int32), header = hdul_old["SCI"].header, name = "M_MASK")
            bkgd_hdu = fits.ImageHDU(bkgd.astype(np.float32), header = hdul_old["SCI"].header, name = "BACKGROUND")
            bkgd_rms_hdu = fits.ImageHDU(bkg_rms.astype(np.float32), header = hdul_old["SCI"].header, name = "BACKGROUND_RMS")


            hdul.append(bkgsub_hdu)
            hdul.append(merge_mask_hdu)
            hdul.append(bkgd_hdu)
            hdul.append(bkgd_rms_hdu)

            hdul.writeto(output_path, overwrite = True)



        print(f"Result stored in {output_path}")
        end_time = datetime.now()
        print(f"Duration is {end_time - start_time}")


    def merge_mask(self, hdul_files, output_path):
        mmask = None
        for file in hdul_files:
            with fits.open(file) as hdul:
                bit_mask = hdul["TIERMASK"].data
                header = hdul["SCI"].header
                if mmask is None:
                    mmask = np.left_shift(np.right_shift(bit_mask, 1), 1)
                else:
                    mmask = mmask | np.left_shift(np.right_shift(bit_mask, 1), 1) # border mask is bit 1.
        mmask = mmask.astype(np.int32)
        hdu = fits.ImageHDU(mmask, header =header, name = "Merged_mask")
        hdu.writeto(output_path, overwrite = True)
    

    def subtract_with_broadermask(self, filepath, mmask_path, output_path):


        with fits.open(filepath) as hdul:
            broader = hdul["TIERMASK"].data == 1
            sci = hdul["SCI"].data
            wcs = WCS(hdul["SCI"].header)


        mmask = fits.getdata(mmask_path)
        source_mask = broader | mmask
        mask = source_mask != 0

        bkg = Background2D(sci,
                    box_size = 10,
                    sigma_clip = astrostats.SigmaClip(sigma=3),
                    filter_size = 5,
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = 90,
                    mask = mask,
                    interpolator = BkgIDWInterpolator())
        bkg_sub = sci - bkg.background

        hdul_new = fits.HDUList()
        msub_sci_hdu = fits.ImageHDU(bkg_sub, header = wcs.to_header(), name = "M-BKGSUB")
        bkg_hdu = fits.ImageHDU(bkg.background, header = wcs.to_header(), name = "M-BKG")
        mask_hdu = fits.ImageHDU(source_mask.astype(np.int8), header = wcs.to_header(), name = "MMask")
        hdul_new.append(msub_sci_hdu)
        hdul_new.append(bkg_hdu)
        hdul_new.append(mask_hdu)

        hdul_new.writeto(output_path, overwrite = True)
        print(f"stored in {output_path}")









        



        
