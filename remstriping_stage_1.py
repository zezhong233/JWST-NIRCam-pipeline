from concurrent.futures import ProcessPoolExecutor 
from astropy.io import fits
from remstriping import striping_noise
from glob import glob
from math import floor
import numpy as np
import os

os.environ['CRDS_PATH']='/home/zhongyi/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"


def striping_worker(args):
    stage1_dir, rate = args
    MASKTHRESH = 0.8
    base_index = os.path.basename(rate).split("_rate.fits")[0]
    pre_1f_name = rate.replace("_rate.fits", "_rate_pre1f.fits")   
    
    with fits.open(rate, mode = "update") as hdul:#update is important.
        sci = hdul["SCI"].data
        sci[np.isnan(sci)] = 0
        hdul["SCI"].data = sci
        hdul.flush()
    print("--Start to do 1/f noise subtration for {0}--".format(rate))
    sn = striping_noise(INPUTDIR=stage1_dir, OUTPUTDIR=stage1_dir, MASKTHRESH=MASKTHRESH) #for bcg field, 0 is recommend. for other condition, 0.8 is better.
    sn.measure_striping(rate, pre_1f_name, save_patterns = False, apply_flat = True)
   

    with fits.open(rate, mode = "update") as hdul:#update is important.
        sci = hdul["SCI"].data
        sci[sci == 0] = np.nan
        hdul["SCI"].data = sci
        hdul.flush()

    with fits.open(pre_1f_name, mode = "update") as hdul:#update is important.
        sci = hdul["SCI"].data
        sci[sci == 0] = np.nan
        hdul["SCI"].data = sci
        hdul.flush()
    return "--Finish 1/f noise subtraction for {0}_rate.fits--".format(base_index)







if __name__ == "__main__":
    rates = sorted(glob("/Data/JWST_ZY/f200w/stage1/*rate.fits"))
    n_maxworker = os.cpu_count()
    stage1_dir = "/Data/JWST_ZY/f200w/stage1"
    tasks = [(stage1_dir, rate) for rate in rates]
    with ProcessPoolExecutor(max_workers=floor(n_maxworker / 2)) as pool:
        for res in pool.map(striping_worker, tasks):
            print(res)
    print("finished")

