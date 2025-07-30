import os
os.environ['CRDS_PATH']='/home/zezhong/work/ImageReduction_Pipeline/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"
import sys
import numpy as np
import math
from glob import glob
from astropy.io import fits
from snowball_run_pipeline import detector1_with_snowball_correction
# from wispsub import wispsub
from subtract_wisp import wisp_class
from remstriping import striping_noise 
from jwst.pipeline import Image2Pipeline
from jwst.pipeline import Image3Pipeline 
from jwst.resample.resample_step import ResampleStep
from jwst.tweakreg.tweakreg_step import TweakRegStep
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
from jwst.skymatch.skymatch_step import SkyMatchStep
from skywcsvar import sky_wcs_var_class        
from jwst.associations.asn_from_list import asn_from_list
from collections import defaultdict
from dataclasses import dataclass
from mosaic_bkg_sub import background_and_tiermask
from pathlib import Path

'''
Find a new wisp subtraction algorithm, looks better.
But needs to change the structure of pipeline, which puts the wisp and 1overf noise into stage2.
And an optional choise is using  LW image, which isn't affected by wisp noise, to do source detection.
Besides, given the speed of stage1, I choose multiPoolProcess to do stage1 and stage2.
-------------------------------------------------------------------------------------------------------
Zhongyi Zhang 2025/07/15
'''

@dataclass
class pipeline():
    lw_dir:     str = "."
    asn_dir:    str = "."
    wisp_dir:   str = "."
    stage0_dir: str = "."
    stage1_dir: str = "."
    stage2_dir: str = "."
    stage3_dir: str = "."
    mosaic_dir: str = "."


    def stage1(self, uncalfile):
        
        base_index = os.path.basename(uncalfile).split("_uncal.fits")[0]
        #run stage1 and snowball
        detector1_with_snowball_correction(base_index, input_dir = self.stage0_dir, output_dir = self.stage1_dir, maxcores = "quarter")
        #base_index_0_*.fits is rate file
        #base_index_1_*.fits is rateints file

        #change the name.
        rampfit = os.path.join(self.stage1_dir, "%s_0_rampfitstep.fits"%base_index)
        rampintsfit = os.path.join(self.stage1_dir, "%s_1_rampfitstep.fits"%base_index)
        rate = os.path.join(self.stage1_dir, "%s_rate.fits"%base_index)
        rateints = os.path.join(self.stage1_dir, "%s_rateints.fits"%base_index)
        os.rename(rampfit, rate)        
        os.rename(rampintsfit, rateints)
        print("------------------------------------Finish jwst.Detector1Pipeline for {0}_uncal.fits--------------------------------------".format(base_index))



    def stage2(self, ratefile, stage2_pipeline = True, wisp = True, one_over_noise = True):

        '''
        ratefile: str
            relative path to *rate.fits, e.g. jw01324001001_02201_00001_nrca3_rate.fits
        '''

        base_index = os.path.basename(ratefile).split("_rate.fits")[0]
        calfile = os.path.join(self.stage2_dir, base_index + "_cal.fits") #absolute filepath 
        # print(calfile)
        if stage2_pipeline:
            print("start executing stage2")
            Image2Pipeline.call(ratefile,output_dir = self.stage2_dir ,steps = {'bkg_subtract':{'skip':False}, 'resample':{'skip':True}}) 
            print("------------------------------------Finish jwst.Image2Pipeline for {0}_rate.fits--------------------------------------".format(base_index))


        with fits.open(calfile, mode = "update") as hdul:
            sci = hdul["SCI"].data
            sci[np.where(np.isnan(sci))] = 0
            hdul["SCI"].data = sci
            hdul.flush()

        if wisp:#will do wisp subtraction 
            sw = wisp_class()
            sw.wisp_dir = self.wisp_dir
            sw.lw_dir = self.lw_dir
            sw.process_file(f = calfile, seg_from_lw = False, save_segmap = False)
            print("------------------------------------Finish wisp_subtraction for {0}_cal.fits--------------------------------------".format(base_index))
            #output: *cal.fits
            #origin: *cal_prewisp.fits

        if one_over_noise:
            stripfile = calfile
            pre1f = stripfile.replace('cal.fits', 'cal_pre1f.fits')  
            striping = striping_noise(self.stage2_dir, self.stage2_dir) 
            striping.measure_striping(stripfile, pre1f, save_patterns = False, apply_flat=False) #don't do flat field. 
            #don't save horizonal and vertical pattern 
            print("------------------------------------Finish 1/f stripping pattern subtraction for {0}_cal.fits--------------------------------------".format(base_index))





    def asn_creation(self,in_suffix, out_suffix , input_dir = "None", output_dir = "None", type:{"single","multiple"} = "single", multi_angles = False):
        '''
        Auxiliary function to generate association file in asdf syntax.
        -----------------------------------------------------------------
        parameters:
        ------------
        in_suffix: str
            suffix of those files to be collected.
        out_suffix: 
            suffix of output .asdf file.
        type:str
            - "single": Assuming there is only a single band in input_dir.
            - "multiple": multiple band.
        
        ------------
        return:
            type == "single"  - fil: str
                filter of this directory.
        '''
        if input_dir == "None":
            input_dir = self.stage3_dir
        if output_dir == "None":
            output_dir = self.stage3_dir

        if type == "single":
            files = sorted(glob("{0}/*{1}.fits".format(input_dir, in_suffix)))
            with fits.open(files[0]) as hdul:
                fil = hdul[0].header["FILTER"]

            if multi_angles:
                position_angle = defaultdict(list)
                for file in files:
                    with fits.open(file) as hdul:
                        hea = hdul[1].header
                        PA_V3 = f"{hea["PA_V3"]:.1f}"
                    position_angle[PA_V3].append(file)
                for angle in position_angle.keys():
                    match_asn = asn_from_list(position_angle[angle], product_name = "nircam_{0}_{1}".format(fil,angle))
                    json = "{0}/nircam_{1}_{2}_{3}.json".format(output_dir, fil, out_suffix,angle)
                    with open(json, "w") as jsonfile:
                        name, serialized = match_asn.dump(format = "json")
                        jsonfile.write(serialized)
                return fil, position_angle
            else:
                lists = files
                match_asn = asn_from_list(lists, product_name = "nircam_%s"%fil)
                json = "{0}/nircam_{1}_{2}.json".format(output_dir, fil, out_suffix)
                with open(json, 'w') as jsonfile:
                    name, serialized = match_asn.dump(format = "json")
                    jsonfile.write(serialized)  
                return fil

        if type == "multiple":
            files = sorted(glob("{0}/*{1}.fits".format(input_dir, in_suffix)))
            dic_fits = defaultdict(list)
            for file in files:
                with fits.open(file) as hdul:
                    fil = hdul[0].header["FILTER"]
                dic_fits[fil].append(file)#按波段分类
            for filter, lists in dic_fits.items():
                match_asn = asn_from_list(lists, product_name = "nircam_%s"%filter)
                json = "{0}/nircam_{1}_{2}.json".format(output_dir, filter, out_suffix)
                with open(json, 'w') as jsonfile:
                    name, serialized = match_asn.dump(format = "json")
                    jsonfile.write(serialized)

    def stage3_part1(self, input_dir = "None", asn_dir = "None",use_custom_catalogs = False,rel_catfile = "", 
                            abs_refcat = "GAIADR3",  update_wcs = True, skymatch = True,outlier_detection = True, sky_wcs_var = True):

        '''
        Do JWST Image3Pipeline's single exposure part, including tweakreg, sky_match, outlier_detection.
        ---------------------------------------------------------------------------------------------------------------
        parameters:
        -------------
        input_dir: str
            directory where store *cal.fits file.
        asn_dir: str
            directory where store the association file.
        use_custom_catalogs: boolean
            whether using user_provided source catalog, If False, the code will automatically create catalogs in each exposure 
            following algorithm photutils.detection.DAOStarFinder. 
        rel_catfile: str
            user provided relative source catalog's. If use_custom_catalogs is False, then it will be ignored.
        abs_refcat: str
            Absolute source catalog
            Can be provided by user. Or  "GAIADR1", "GAIADR2" and "GAIADR3"
            Format can be found here: https://jwst-pipeline.readthedocs.io/en/latest/jwst/tweakreg/README.html#step-arguments
        update_wcs, skymatch, outlier_detection, sky_wcs_var: boolean
            whether to run this step. 
            Only for test, all true by default.
        '''

        #If input_dir or asn_dir is "None", use default address.
        if input_dir == "None":
            input_dir = self.stage2_dir
        if asn_dir == "None":
            asn_dir = self.asn_dir
    
        if update_wcs:
            #generate association file for tweakreg step.
            fil = self.asn_creation(in_suffix = "cal", out_suffix = "tweakreg",input_dir=input_dir, output_dir=asn_dir)
            json_t = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "tweakreg"))
            #run jwst tweakreg step 
            TweakRegStep.call(json_t, use_custom_catalogs = use_custom_catalogs,catfile = rel_catfile,  catalog_format = "ecsv",
                                        brightest = 400, npixels = 20, contrast = 0.005, nclip = 5, sigma = 3, 
                                        abs_refcat = abs_refcat, abs_fitgeometry = "general", save_results = True, 
                                        output_dir = self.stage3_dir, suffix = "tweakreg"
                                    )
        #do sky_match(temporarily can't subtract the bkg) and outlier detection 
        if skymatch:
            #generate association file for skymatch step.
            fil = self.asn_creation(in_suffix = "tweakreg", out_suffix = "sky_match", output_dir=asn_dir)
            json_s = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "sky_match"))
            #run jwst skymatch step
            SkyMatchStep.call(json_s,output_dir = self.stage3_dir, suffix = "skymatch", save_results = True, skymethod = "local", 
                              output_use_index = False, output_use_model = True)
            
        if outlier_detection:
            #generate association file for skymatch step.
            fil = self.asn_creation(in_suffix = "skymatch", out_suffix = "outlier", output_dir=asn_dir)
            json_o = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "outlier"))
            #run jwst jwst outlierDetection
            OutlierDetectionStep.call(json_o, output_dir = self.stage3_dir, suffix = "crf",save_results = True,
                                       output_use_index = False, output_use_model = True)

            #rename the output file, for simpleness. 
            crfs = sorted(glob("{0}/*_crf.fits".format(self.stage3_dir)))   
            for crf in crfs: 
                p = Path(crf) 
                p.rename(p.with_name(p.stem.replace("skymatch_a3001_crf", "_a3001_crf") + p.suffix)) 
            crfs = sorted(glob("{0}/*_crf.fits".format(self.stage3_dir)))

            for crf in crfs:
                with fits.open(crf, mode = "update") as hdul:
                    r_noise = hdul[6].data
                    r_noise[np.where(np.isnan(r_noise))] = 0
                    hdul["VAR_RNOISE"].data = r_noise
                    hdul.flush()

        #correct for the variance map(for 0 value in bad area which don't contribute varicance, so set value in this area in inf)
        #  and sky_match(for small overlapping area, minus a pedestal value.)
        if sky_wcs_var:
            crfs = sorted(glob("{0}/*_crf.fits".format(self.stage3_dir)))  
            for crf in crfs:
                match_file = crf.replace("crf.fits", "match.fits")
                if os.path.exists(match_file):
                    print(f"The sky_wcs_var step has been implied for {os.path.basename(crf)}, so skip it.")
                    continue
                else:
                    swv = sky_wcs_var_class()
                    swv.INPUTDIR = self.stage3_dir
                    swv.OUTPUTDIR = self.stage3_dir
                    swv.MASKDIR = self.stage2_dir
                    swv.process(os.path.basename(crf))    
            #By default, get _a3001_match.fits, which is ready to do mosaic. and _a3001_bkgsub_1.fits.

    def cal_rotation(self, h):
        
        pcs = np.array([[ h['PC1_1'],  h['PC1_2']], [h['PC2_1'], h['PC2_2']]])
        cd = np.array([[h['CDELT1'], 0],[0, h['CDELT2']]])
        cd_rot=np.dot(pcs,cd)
        w1 = cd_rot[0,0]
        w2 = cd_rot[1,0]
        rotation = math.atan(-w2/w1)/math.pi*180
        return rotation




    def stage3_part2(self, crpix:list = None, crval:list = None, rotation:float = None,
                        pixfrac:float= None, pixel_scale:float = None, outputshape:list = None, 
                        asn_dir:str = ".",
                        weight_type:str = "ivm",
                        multi_angles = False):
        '''
        Do mosaic creation (resample step) and final background subtraction.
        '''

        if multi_angles:   
             
            fil, ang_fit_dic= self.asn_creation(in_suffix = "a3001_match", out_suffix = "mosaic", input_dir = self.stage3_dir,output_dir = asn_dir, multi_angles=True)
            for angle in list(ang_fit_dic.keys()):

                json_m = os.path.join(asn_dir, "nircam_{}_{}_{}.json".format(fil, "mosaic", angle))
                mosaic = ResampleStep.call(json_m, crpix = crpix, crval = crval,rotation = rotation, #rotation will be ignored if pixel_scale is given, and it will adopt default value.
                                        output_dir = self.mosaic_dir, save_results = True, 
                                        pixfrac = pixfrac, pixel_scale = pixel_scale, output_shape = outputshape, 
                                        weight_type= weight_type)
                resample = "{0}/nircam_{1}_mosaic_{2}_resample.fits".format(self.mosaic_dir, fil, angle)
                suffix = "bkg_sub" 
                background_and_tiermask(resample, suffix, resample.split("/nircam")[0], self.mosaic_dir)

        else:

            fil = self.asn_creation(in_suffix = "a3001_match", out_suffix = "mosaic", input_dir = self.stage3_dir,output_dir = asn_dir, multi_angles = False)
            json_m = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "mosaic"))
            mosaic = ResampleStep.call(json_m, crpix = crpix, crval = crval,rotation = rotation, #rotation will be ignored if pixel_scale is given, and it will adopt default value.
                        output_dir = self.mosaic_dir, save_results = True, 
                        pixfrac = pixfrac, pixel_scale = pixel_scale, output_shape = outputshape, 
                        weight_type= weight_type)
            
            resample = "{0}/nircam_{1}_mosaic_resample.fits".format(self.mosaic_dir, fil)
            suffix = "bkg_sub" 
            background_and_tiermask(resample, suffix, resample.split("/nircam")[0], self.mosaic_dir)


            # match_fits = ang_fit_dic[angle][0]
            # base = os.path.basename(match_fits)
            # base = os.path.join(self.stage0_dir, base)
            # uncal = base.replace("a3001_match.fits", "uncal.fits")
            # header = fits.getheader(uncal, 1)
            # if rotation is None:
            #     rotation = self.cal_rotation(header)
            # json_m = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "mosaic")) 