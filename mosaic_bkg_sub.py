### ############
#2025.06.20
#Zhangzhongyi 
#Follow Ceers' method to do another bkgsub for doc's pipeline to check if result is basicly same.
#The result is promising!
#Result is save in /home/zezhong/work/JWST_data_reduction/9.bkg_sub/output /drizzle_F***W_bkg_sub.fits
### ###########


from configparser import ConfigParser
from astropy.io import fits
import background_subtraction
import os 


def background_and_tiermask(image, bkg_suffix, inputdir, outputpath):
    #subtract the background using a single four-tier mask
    cfg = ConfigParser()
    cfg.read("/home/zezhong/work/Final_ImageReduction_Pipeline/pipeline/jwst/mosaic_background.cfg")
    options = cfg.options("nircam")
    parameters = {}
    # get the paramters
    for option in options:
        parameters[option] = cfg.get("nircam", option)
    bs = background_subtraction.SubtractBackground()
    bs.suffix = bkg_suffix 
    bs.replace_sci = True
    bs.bg_box_size = int(parameters['bg_box_size'])
    bs.bg_filter_size = int(parameters['bg_filter_size'])
    bs.ring_radius_in = int(parameters['ring_radius_in'])
    bs.ring_width = int(parameters['ring_width'])
    tier_dilate_size = [int(x) for x in parameters['tier_dilate_size'].split(',')]
    bs.tier_dilate_size = tier_dilate_size
    tier_nsigma = [float(x) for x in parameters['tier_nsigma'].split(',')]
    bs.tier_nsigma = tier_nsigma
    tier_npixels = [int(x) for x in parameters['tier_npixels'].split(',')]
    bs.tier_npixels = tier_npixels

    bs.do_background_subtraction(inputdir,os.path.basename(image),outputpath)

