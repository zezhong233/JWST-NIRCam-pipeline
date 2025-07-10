import os
os.environ['CRDS_PATH']='your crds directory'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"

from pipeline_c import pipeline
from glob import glob
from astropy.io import fits

stage0_dir = "your stage0 directory"
stage1_dir = "your stage1 directory"
stage2_dir = "your stage2 directory"
stage3_dir = "your stage3 directory"
mosaic_dir = "your mosaic directory"

def run():
    pl = pipeline()
    pl.stage0_dir = stage0_dir
    pl.stage1_dir = stage1_dir
    pl.stage2_dir = stage2_dir
    pl.stage3_dir = stage3_dir
    pl.mosaic_dir = mosaic_dir
    
#stage1_wf
    uncal_files = sorted(glob("./*uncal.fits"))
    for uncal_file in uncal_files:
        pl.stage1_wf(uncal_file)

#stage2
    ratefiles = sorted(glob("%s/*rate.fits"%stage1_dir)) 
    for ratefile in ratefiles: 
        pl.stage2_ff(ratefile) 

#stage3_part1 
    abs_refcat = "GAIADR3"
    asn_dir = "./asn" #your asn files directory
    pl.stage3_part1(asn_dir = asn_dir, abs_refcat = abs_refcat, update_wcs = True ,skymatch =  True, outlier_detection = True, sky_wcs_var = True)

#stage3_part2 
    asn_dir = "./asn"
    ref_fits = "./*mosaic.fits"
    header = fits.getheader(ref_fits, 0)
    crpix = [header["CRPIX1"] - 1,header["CRPIX2"] - 1] 
    crval = [header["CRVAL1"], header["CRVAL2"]]
    pix_frac = header["PIXFRAC"]
    pix_scale = 0.02 #adjust 
    pl.stage3_part2(asn_dir = asn_dir, crpix = crpix, crval = crval, pix_frac = pix_frac, pix_scale = pix_scale) 


if __name__ == "__main__": 
    run() 