from pipeline_new import pipeline 
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from datetime import datetime 
import os
from math import floor

filter = "200"
ID = "1324"
wisp_dir = "/home/zezhong/work/ImageReduction_Pipeline/CRDS/wisp_template_ver3"
sci_dir = f"/mnt/data/UNCOVER_grism/Project/JWST_NIRCam_{ID}/F{filter}W"
stage0_dir = os.path.join(sci_dir, "uncal")
stage1_dir = os.path.join(sci_dir, "stage1_new")
stage2_dir = os.path.join(sci_dir, "stage2_new")
stage3_dir = os.path.join(sci_dir, "stage3_new")
mosaic_dir = os.path.join(sci_dir, "mosaic_new")
asn_dir = os.path.join(sci_dir, "asn_new")
lw_dir = os.path.join(sci_dir,"lw_new")#optional 

def run_stage1(args):
    '''
    args: 二元元组    
    '''
    cfg, uncalfile = args
    pl = pipeline(**cfg)
    pl.stage1(uncalfile = uncalfile, )
    return f"{uncalfile}, Done."


def run_stage2(arg):
    cfg, ratefile = arg
    pl = pipeline(**cfg)
    pl.stage2(ratefile = ratefile, stage2_pipeline=True, wisp = True, one_over_noise = True)
    return f"{ratefile}, Done"


def run_stage3(cfg):

    pl = pipeline(**cfg)
    pl.stage3_part1(update_wcs = True, skymatch = True, outlier_detection = True,  sky_wcs_var = True)
    return "stage3 pipeline except drizzle Done."

def run_resample(cfg):
    pl = pipeline(**cfg)
    if filter in ["090", "115","150","200"]:
        pixel_scale = 0.02
    else:
        pixel_scale = 0.04
    pl.stage3_part2(pixfrac = 1.0, pixel_scale = pixel_scale, multi_angles = False)


def main(stage1, stage2, stage3, resample):

    n_maxworker = os.cpu_count()

    cfg = dict(
        lw_dir= lw_dir,
        asn_dir = asn_dir,
        wisp_dir=wisp_dir,
        stage0_dir=stage0_dir,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        stage3_dir = stage3_dir,
        mosaic_dir = mosaic_dir,
    )

    #收集要处理的文件 
    if stage1:
        start = datetime.now()
        files_stage1 = sorted(glob(os.path.join(cfg["stage0_dir"], "*uncal.fits")))
        tasks_stage1 = [(cfg, f) for f in files_stage1]
        
        with ProcessPoolExecutor(max_workers=8) as pool:
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

    if resample:
        start = datetime.now()
        run_resample(cfg)
        end = datetime.now()
        print("-------------------------------------Resample During time is {}.------------------------------------".format(end-start))





if __name__ == "__main__":
    main(stage1 = False, stage2 = False, stage3 = True, resample = True) 


