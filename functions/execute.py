import os
os.environ['CRDS_PATH']='/home/zezhong/work/ImageReduction_Pipeline/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"

from pipeline_c import pipeline
from glob import glob
from astropy.io import fits

filter = "200"
ID = "2561"
stage0_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/uncal"
stage1_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/stage1"
stage2_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/stage2"
stage3_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/stage3"
mosaic_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/mosaic"

def run():
    pl = pipeline()
    pl.stage0_dir = stage0_dir
    pl.stage1_dir = stage1_dir
    pl.stage2_dir = stage2_dir
    pl.stage3_dir = stage3_dir
    pl.mosaic_dir = mosaic_dir
    
#stage1_wf
    uncal_files = sorted(glob(f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/uncal/*uncal.fits"))
    for uncal_file in uncal_files:
        pl.stage1_wf(uncal_file, stage1_and_snowball = False, wisp = False)

#stage2
    ratefiles = sorted(glob("%s/*rate.fits"%stage1_dir)) 
    for ratefile in ratefiles: 
        pl.stage2_ff(ratefile) 

#stage3_part1 
    abs_refcat = "GAIADR3"
    asn_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/asn"
    pl.stage3_part1(asn_dir = asn_dir, abs_refcat = abs_refcat, update_wcs = True ,skymatch =  True, outlier_detection = True, sky_wcs_var = True)
    #需要做skymatch吗？

#stage3_part2 
    file = f"/mnt/data/JWST/UNCOVER/NIRCAM_Primary/abell2744clu-grizli-v5.4-f{filter}w-clear_drc_sci.fits"
    with fits.open(file) as hdul:
        # hdul.info()
        header = hdul[0].header
        crpix = [header["CRPIX1"] - 1,header["CRPIX2"] - 1] 
        crval = [header["CRVAL1"], header["CRVAL2"]]

        if filter in ["090", "115", "150", "200"]:#SW
            pix_scale = 0.02
        else:#LW, ["277", "356", "410", "444"]
            pix_scale = 0.04

        pix_frac = header["PIXFRAC"]
    asn_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W/asn"
    # pl.stage3_part2(asn_dir = asn_dir, crpix = crpix, crval = crval, 
    #                 pixfrac = pix_frac, pixel_scale = pix_scale) 
    pl.stage3_part2(asn_dir = asn_dir, pixfrac = pix_frac, pixel_scale = pix_scale)

if __name__ == "__main__": 
    run() 