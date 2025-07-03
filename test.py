import os
os.environ['CRDS_PATH']='/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"

from pipeline_c import pipeline
from glob import glob


stage1_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/stage1"
stage2_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/stage2"
stage3_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/stage3"
mosaic_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/mosaic"

def run():
    pl = pipeline()
    pl.stage1_dir = stage1_dir
    pl.stage2_dir = stage2_dir
    pl.stage3_dir = stage3_dir
    pl.mosaic_dir = mosaic_dir
# #stage1_wf
#     uncal_files = sorted(glob("/mnt/data/CEERS/NIRCAM/uncals/F200W/*uncal.fits"))
#     for uncal_file in uncal_files:
#         pl.stage1_wf(uncal_file,stage1_and_snowball = True,wisp = True, striping = True)
# #stage2
#     ratefiles = sorted(glob("%s/*rate.fits"%stage1_dir)) 
#     for ratefile in ratefiles: 
#         pl.stage2_ff(ratefile) 
#stage3_part1 
    abs_refcat = "/mnt/data/CEERS/NIRCAM/align/0511_ref_cat/cat_ref_F200W.ecsv"
    asn_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/pipeline/ceers/asns"
    pl.stage3_part1(asn_dir = asn_dir, abs_refcat = abs_refcat, update_wcs = True ,skymatch = True, outlier_detection = True, sky_wcs_var = True)
#stage3_part2 
    asn_dir = "/home/zezhong/work/Final_ImageReduction_Pipeline/pipeline/asns"
    pl.stage3_part2(asn_dir = asn_dir, make_mosaic = True, final_bkgsub = True) 

if __name__ == "__main__": 
    run() 