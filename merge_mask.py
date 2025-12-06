
from astropy.io import fits
from mine_utils import show_image,get_files
from astropy.wcs import WCS
from reproject import reproject_interp
from concurrent.futures import ProcessPoolExecutor 
import numpy as np
import sys 


import os
os.chdir("/RS2423/JWST/BLAGN_legacy/CEERS_PID_1345_NIRCam")


cpu_count = os.cpu_count()

bands = ["115W", "150W", "200W", "277W", "356W", "410M", "444W"]

def save_fits(filename, data, header):
    fits.writeto(filename, data, header, overwrite = True)
    print(f"data has been stored in {filename}")


def reproject_mask(paths, i, tar_ID, parent_dir):

    path_i = paths[i]
    path_tar = paths[tar_ID]
    with fits.open(path_i) as hdul_i, fits.open(path_tar) as hdul_tar:
        mask_i = hdul_i["TIERMASK"].data.astype(bool)
        header_i = hdul_i["TIERMASK"].header
        mask_tar = hdul_tar["TIERMASK"].data.astype(bool)
        header_tar = hdul_tar["TIERMASK"].header
    
    ori_wcs = WCS(header_i)
    tar_wcs = WCS(header_tar)

    mask_interp, _ = reproject_interp((mask_i, ori_wcs), tar_wcs, shape_out = mask_tar.shape)
    output_dir = f"{parent_dir}/{bands[tar_ID]}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    save_fits(f"{output_dir}/mask_{bands[i]}_{bands[tar_ID]}_interp.fits", data = mask_interp, header = tar_wcs.to_header())
    return f"{bands[i]}_{bands[tar_ID]} finished."



def get_masks(paths):
    masks = []
    wcss = []
    for path in paths:
        with fits.open(path) as hdul:
            tier_mask = hdul["TIERMASK"].data
            header = hdul["TIERMASK"].header
        wcs = WCS(header)
        mask_bool = tier_mask.astype(bool)
        masks.append(mask_bool)
        wcss.append(wcs)
    return masks, wcss

def merge_mask(mask_paths,  in_band, header = None, save = True, parent_dir = None):

    masks = []
    # 在子进程内部读取文件
    for i, path in enumerate(mask_paths):
        with fits.open(path) as hdul:
            mask = hdul[0].data
            # 如果没有传入 header，读取第一个文件的 header 作为参考
            if header is None and i == 0:
                header = hdul[0].header
            masks.append(mask)

    masks_bool = [np.asarray(m, dtype=bool) for m in masks]
    merged = masks_bool[0].copy()
    for m in masks_bool[1:]:
        np.logical_or(merged, m, out=merged)
    output_dir = f"{parent_dir}/{in_band}"
    if save:
        save_fits(f"{output_dir}/merged_{in_band}.fits", data = merged.astype(int), header = header)
    # return merged   
    return f"{in_band} mask has been merged"

if __name__ == "__main__":

    POINTING_ID = 4

    paths = get_files(f"Pointing_{POINTING_ID}/F*/resample/bkgsub.fits")

    # masks, wcss = get_masks(paths)

    bands = ["115W", "150W", "200W", "277W", "356W", "410M", "444W"]
    assert len(bands) == len(paths), "shape isn't same."

    parent_dir = f"Pointing_{POINTING_ID}/M_DATA"

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


    all_futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(cpu_count / 2))) as executor:
        for tar_ID in [2,3,4,5,6]:
        # for tar_ID in [0,1]:
            for i in range(len(bands)):
                f = executor.submit(reproject_mask,
                                    paths,i, tar_ID, parent_dir)
                all_futures.append(f)

    for f in all_futures:
        f.result()


    merge_futures = []

    with ProcessPoolExecutor(max_workers=max(1, int(cpu_count / 2))) as executor_2:

        for in_band in bands:
            print("hahaha")
            mask_paths = get_files(f"Pointing_{POINTING_ID}/M_DATA/{in_band}/*interp.fits")
            if len(mask_paths) < 7:
                break
            f = executor_2.submit(merge_mask,
                                  mask_paths, in_band, header = None, parent_dir = parent_dir)
            # f.result()
            merge_futures.append(f)
    for f in merge_futures:
        f.result()
