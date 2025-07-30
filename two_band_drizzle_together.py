from pipeline_newnew import asn_creation, cal_rotation
from jwst.resample.resample_step import ResampleStep
from astropy.io import fits
from glob import glob
from mosaic_bkg_sub import background_and_tiermask
import os
if False:
    dir_1324 = "/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_1324/F200W/stage3_new"
    files_1324 = sorted(glob(os.path.join(dir_1324, "*a3001_match.fits")))
    dir_2561 = "/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_2561/F200W/stage3_new"
    files_2561 = sorted(glob(os.path.join(dir_2561,"*a3001_match.fits")))

    files = files_1324 + files_2561
    output_dir = "/mnt/data/UNCOVER_grism/Project/asn_multi_survey"
    p_1324 = asn_creation(in_suffix = "a3001_match.fits", out_suffix = "1324_2561_F200W", input_dir = ".",files = files, output_dir=output_dir)

resample_and_bkgsub = True

if resample_and_bkgsub:

    json_m = "/mnt/data/UNCOVER_grism/Project/asn_multi_survey/nircam_F200W_1324_2561_F200W.json"
    mosaic_dir = "/mnt/data/UNCOVER_grism/Project/mosaic_multi_survey"
    UNCOVER_MOSAIC = "/mnt/data/UNCOVER_grism/UNCOVER_DR3/UNCOVER_NIRCam_F200W_bkgsub_sci.fits"
    header = fits.getheader(UNCOVER_MOSAIC)
    crpix = [header["CRPIX1"] - 1,header["CRPIX2"] - 1] 
    crval = [header["CRVAL1"], header["CRVAL2"]]
    pix_frac = header["PIXFRAC"]
    fil = "200"
    if fil in ["090", "115","150","200"]:
        pixel_scale = 0.02
    else:
        pixel_scale = 0.04
    rotation = cal_rotation(header)
    mosaic = ResampleStep.call(json_m, crpix = None, crval = None,rotation = rotation, #rotation will be ignored if pixel_scale is given, and it will adopt default value.
                output_dir = mosaic_dir, save_results = True, 
                pixfrac = pix_frac, pixel_scale = pixel_scale, output_shape = None, 
                weight_type= "ivm", in_memory = False)
    
    resample = "{0}/nircam_{1}_mosaic_resample.fits".format(mosaic_dir, fil)
    suffix = "bkg_sub" 
    background_and_tiermask(resample, suffix, resample.split("/nircam")[0], mosaic_dir)


if __name__ == "__main__":
    resample_and_bkgsub()
