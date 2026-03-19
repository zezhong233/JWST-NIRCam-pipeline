from pipeline import pipeline 
from pipeline import cal_rotation
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from datetime import datetime 
from astropy.io import fits
import os
from math import floor
import numpy as np

os.environ['CRDS_PATH']='/home/zhongyi/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
# os.environ['CRDS_SERVER_URL'] = "none"
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"


def run_stage1(args):
    '''
    args: 二元元组    
    '''
    cfg, uncalfile = args
    pl = pipeline(**cfg)
    maskthreshold = 0.8
    pl.stage1(uncalfile = uncalfile, stage1_snow = True, stripping = True, MASKTHRESH=maskthreshold) #for density area, 0 is recommended. Others, 0.8 is better.
    return f"{uncalfile}, Done."


def run_stage2(arg):
    cfg, ratefile = arg
    pl = pipeline(**cfg)
    pl.stage2(ratefile = ratefile, stage2=True, wisp = True)
    return f"{ratefile}, Done"


def run_stage3(cfg):

    pl = pipeline(**cfg)
    # abs_refcat = "/Data/BLAGN_legacy/CEERS_EGS_HST_v1.9_cat_radecmag.ecsv" 
    abs_refcat = None
    pl.stage3_part1(use_custom_catalogs=False, abs_refcat = abs_refcat, 
                    update_wcs = True, skymatch = True, outlier_detection = True,  sky_wcs_var = False) #先只做前三步



def run_sky_var(arg):

    cfg, crf = arg
    pl = pipeline(**cfg)
    pl.stage3_part1(update_wcs = False, skymatch = False, outlier_detection = False, sky_wcs_var = True, crf = crf)#做最后一步
    return f"finish for {os.path.basename(crf)}"

def run_resample(cfg, pixel_scale = None, pixel_frac = None,header = None, output_shape = None,output_path = None):

    
    pl = pipeline(**cfg)
    #pix_scale
    if pixel_scale == None:
        pixel_scale = 0.02
    if pixel_frac == None:
        pixel_frac = 0.75
    rotation = None
    if header is not None:
        rotation = cal_rotation(header, pixel_scale)
    pl.resample(outputshape = output_shape,rotation = rotation,
                           pixfrac = pixel_frac, pixel_scale = pixel_scale,output_path = output_path)
    return "finish resample."

def run_bkgsub(cfg, pixel_scale):

    factor = 0.03/pixel_scale
    pl = pipeline(**cfg)
    pl.bkgsub(factor = factor)
    return "finish bkgsubing"


def process_band(cfg, stage1 = False, stage2 = False, stage3 = False, sky_var = False, 
                 resample = False, bkgsub = False, drizzle_pixscl = None, drizzle_pixfrac = None, output_shape = None):
    """
    Main pipeline processing function for a single band.
    """
    print(f"--- Starting processing for filter: {cfg['filter']} ---")

    n_maxworker = os.cpu_count() 
    workers = 8
    if stage1:
        start = datetime.now()
        files_stage1 = sorted(glob(os.path.join(cfg["stage0_dir"], "*uncal.fits")))
        n_files_uncal = len(files_stage1)
        print("stage1 files' length is:", n_files_uncal)
        tasks_stage1 = [(cfg, f) for f in files_stage1]
        n_cpu_used = floor(n_maxworker / 2)
        # workers = min(n_files_uncal, n_cpu_used)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(run_stage1, tasks_stage1):
                print(res)
        end = datetime.now()
        print(f"--Stage1 for F{cfg['filter']} took {end-start}.--")

    if stage2:
        files_stage2 = sorted(glob(os.path.join(cfg["stage1_dir"], "*rate.fits")))
        tasks_stage2 = [(cfg, f) for f in files_stage2]
        start = datetime.now()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(run_stage2, tasks_stage2):
                print(res)
        end = datetime.now()    # 100files - 20min 
        print(f"--Stage2 for F{cfg['filter']} took {end-start}.--")

    if stage3:
        start = datetime.now()
        run_stage3(cfg)
        end = datetime.now()
        print(f"--Stage3 for F{cfg['filter']} took {end-start}.--")

    if sky_var:
        start = datetime.now()
        files_crfs = sorted(glob(os.path.join(cfg["stage3_dir"], "*a3001_crf.fits")))
        tasks = [(cfg, f) for f in files_crfs]
        with ProcessPoolExecutor(max_workers = workers) as pool:
            for res in pool.map(run_sky_var, tasks):
                print(res)
        end = datetime.now()
        print(f"--sky_wcs_var for F{cfg['filter']} took {end-start}.--")

    if resample:
        start = datetime.now()
        rot_header = fits.getheader(sorted(glob(cfg["stage3_dir"] + "/*a3001_match.fits"))[0], 1)
        run_resample(cfg, pixel_scale = drizzle_pixscl, pixel_frac = drizzle_pixfrac,header = rot_header, output_shape=output_shape)
        end = datetime.now()
        print(f"--Resample for F{cfg['filter']} took {end-start}.--")
    
    if bkgsub:
        start = datetime.now()
        fitsfile = f"nircam_F{cfg['filter']}_mosaic_resample.fits"
        run_bkgsub(cfg, pixel_scale = drizzle_pixscl)
        end = datetime.now()
        print(f"--bkgsub for F{cfg['filter']} took {end-start}.--")
    
    print(f"--- Finished processing for filter: {cfg['filter']} ---")


if __name__ == "__main__":

    filters = ["277W"]
    # filters = ["150W"]

    wisp_dir = "/home/zhongyi/CRDS/wisp_template_ver3"
    for filter in filters:
        # sci_dir = f"/RS2423/JWST/BLAGN_legacy/CEERS_PID_1345_NIRCam/Pointing_{pointing}/F{filter}"
        sci_dir = "/RS2423/JWST/BLAGN_legacy/PRIMER_PID_1837_NIRCam/example/F277W"
        stage0_dir = os.path.join(sci_dir, "uncal") 
        stage1_dir = os.path.join(sci_dir, "stage1")
        stage2_dir = os.path.join(sci_dir, "stage2")
        stage3_dir = os.path.join(sci_dir, "stage3")
        mosaic_dir = os.path.join(sci_dir, "resample")
        asn_dir = os.path.join(sci_dir, "asn")
        lw_dir = os.path.join(sci_dir,"lw")

        cfg = dict(
            lw_dir= lw_dir,
            asn_dir = asn_dir,
            wisp_dir= wisp_dir,
            stage0_dir= stage0_dir,
            stage1_dir= stage1_dir,
            stage2_dir= stage2_dir,
            stage3_dir = stage3_dir,
            mosaic_dir = mosaic_dir,
            filter = filter
        )

        # shape = [8000, 4000]
        process_band(cfg, stage1 = True, stage2=True, stage3 = True, sky_var=True, resample = True, bkgsub = True , drizzle_pixfrac=0.75, drizzle_pixscl=0.04, output_shape=None) 