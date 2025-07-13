### ############
#2025.06.20
#Zhangzhongyi 
#Follow Ceers' method to do another bkgsub for doc's pipeline to check if result is basicly same.
### ###########


from configparser import ConfigParser
from astropy.io import fits
import background_subtraction
import os 

 
def background_and_tiermask(image, bkg_suffix, inputdir, outputpath):
    #subtract the background using a single four-tier mask
    cfg = ConfigParser()
    cfg.read("/home/zezhong/work/ImageReduction_Pipeline/pipelines/nircam_ceers_pipeline/mosaic_background.cfg")
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



if __name__ == "__main__":
    mosaic_dir = "/mnt/data/UNCOVER_grism/Project/JWST_NITCam_1324/F090W/mosaic"
    angles = [43.2, 251.3]
    fil = "F090W"
    for angle in angles:
        resample = "{0}/nircam_{1}_mosaic_{2}_resample.fits".format(mosaic_dir, fil, angle)
        suffix = "bkg_sub" 
        background_and_tiermask(resample, suffix, resample.split("/nircam")[0], mosaic_dir)