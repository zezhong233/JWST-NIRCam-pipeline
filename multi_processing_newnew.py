from pipeline_newnew import pipeline 
from pipeline_newnew import cal_rotation
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from datetime import datetime 
from astropy.io import fits
import os
from math import floor

filter = "200"
ID = "2561"
wisp_dir = "/home/zezhong/work/ImageReduction_Pipeline/CRDS/wisp_template_ver3"
sci_dir = f"/mnt/data/UNCOVER_grism/Project/1324_2561_F200W_2"
stage0_dir = os.path.join(sci_dir, "uncal")
stage1_dir = os.path.join(sci_dir, "stage1")
stage2_dir = os.path.join(sci_dir, "stage2")
stage3_dir = os.path.join(sci_dir, "stage3")
mosaic_dir = os.path.join(sci_dir, "mosaic")
asn_dir = os.path.join(sci_dir, "asn")
lw_dir = os.path.join(sci_dir,"lw")#optional 

cfg = dict(
    lw_dir= lw_dir,
    asn_dir = asn_dir,
    wisp_dir= wisp_dir,
    stage0_dir= stage0_dir,
    stage1_dir= stage1_dir,
    stage2_dir= stage2_dir,
    stage3_dir = stage3_dir,
    mosaic_dir = mosaic_dir,
)

mosaic = f"/mnt/data/UNCOVER_grism/UNCOVER_DR3/mosaic/UNCOVER_NIRCam_F{filter}W_mosaic.fits"

def run_stage1(args):
    '''
    args: 二元元组    
    '''
    cfg, uncalfile = args
    pl = pipeline(**cfg)
    pl.stage1(uncalfile = uncalfile, stage1_snow = True, stripping = True)
    return f"{uncalfile}, Done."


def run_stage2(arg):
    cfg, ratefile = arg
    pl = pipeline(**cfg)
    pl.stage2(ratefile = ratefile, stage2=True, wisp = True)
    return f"{ratefile}, Done"


def run_stage3(cfg):

    pl = pipeline(**cfg)
    # rel_catfile_dir = "GalfitX"
    abs_refcat = "/mnt/data/UNCOVER_grism/UNCOVER_DR3/catalogs/Dey_merged_radec.ecsv" #band?
    pl.stage3_part1(use_custom_catalogs=False, abs_refcat = abs_refcat, 
                    update_wcs = True, skymatch = True, outlier_detection = True,  sky_wcs_var = False)
    # pl.stage3_part1(update_wcs = True, skymatch = True,outlier_detection = True, sky_wcs_var = False)
    return "stage3 pipeline except drizzle Done."


def run_sky_var(arg):

    cfg, crf = arg
    pl = pipeline(**cfg)
    pl.stage3_part1(update_wcs = False, skymatch = False, outlier_detection = False, sky_wcs_var = True, crf = crf)
    return f"finish for {os.path.basename(crf)}"


def run_resample(cfg):
    pl = pipeline(**cfg)
    
    #pix_scale
    if filter in ["090", "115","150","200"]:
        pixel_scale = 0.02
    else:
        pixel_scale = 0.04

    #pixfrac
    header = fits.getheader(mosaic)
    crpix = [header["CRPIX1"] - 1,header["CRPIX2"] - 1] 
    crval = [header["CRVAL1"], header["CRVAL2"]]
    pix_frac = header["PIXFRAC"]
    rotation = cal_rotation(header)

    pl.resample_and_bkgub(outputshape = [28200, 27800],crpix = None, crval = None, rotation = rotation,asn_dir = asn_dir,
                           pixfrac = pix_frac, pixel_scale = pixel_scale, multi_angles = False)

def main(stage1, stage2, stage3, sky_var, resample):

    n_maxworker = os.cpu_count()

    #收集要处理的文件 
    if stage1:
        start = datetime.now()
        files_stage1 = sorted(glob(os.path.join(cfg["stage0_dir"], "*uncal.fits")))
        tasks_stage1 = [(cfg, f) for f in files_stage1]
        
        with ProcessPoolExecutor(max_workers=floor(n_maxworker/2)) as pool:
            for res in pool.map(run_stage1, tasks_stage1):
                print(res)
        end = datetime.now() # 100files - 30 min
        print("-------------------------------------Stage1 During time is {}.------------------------------------".format(end-start))


    if stage2:
        files_stage2 = sorted(glob(os.path.join(cfg["stage1_dir"], "*rate.fits")))
        tasks_stage2 = [(cfg, f) for f in files_stage2]
        start = datetime.now()
        with ProcessPoolExecutor(max_workers=floor(n_maxworker/2)) as pool:
            for res in pool.map(run_stage2, tasks_stage2):
                print(res)
        end = datetime.now()    # 100files - 20min 
        print("-------------------------------------Stage2 During time is {}.------------------------------------".format(end-start))

    if stage3:
        start = datetime.now()
        run_stage3(cfg)
        end = datetime.now()
        print("-------------------------------------Stage3 During time is {}.------------------------------------".format(end-start))



    if sky_var:
        start = datetime.now()
        files_crfs = sorted(glob("/mnt/data/UNCOVER_grism/Project/1324_2561_F200W_2/stage3/*_a3001_crf.fits"))
        tasks = [(cfg, f) for f in files_crfs]
        with ProcessPoolExecutor(max_workers = floor(n_maxworker/2)) as pool:
            for res in pool.map(run_sky_var, tasks):
                print(res)
        end = datetime.now()
        print("-------------------------------------sky_wcs_var During time is {}.------------------------------------".format(end-start))



    if resample:
        start = datetime.now()
        run_resample(cfg)
        end = datetime.now()
        print("-------------------------------------Resample During time is {}.------------------------------------".format(end-start))

if __name__ == "__main__":
    main(stage1 = False, stage2 = False, stage3 = False, sky_var = False, resample = True)  #等着1324一起drizzle 
    # test_crf = "/mnt/data/UNCOVER_grism/Project/1324_2561_F200W_2/stage3/jw01324001001_17201_00001_nrca1_a3001_crf.fits"
    # print(run_sky_var((cfg, test_crf)))
