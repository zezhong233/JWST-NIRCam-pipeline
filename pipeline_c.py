import os
os.environ['CRDS_PATH']='/home/zezhong/work/Final_ImageReduction_Pipeline/Sci_dir/CRDS'  ## specify a PATH to save JWST reference file, more than 20GiB needed
os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu' ## download JWST reference file
os.environ["CRDS_CONTEXT"] = "jwst_1364.pmap"
import sys
import numpy as np
from glob import glob
from astropy.io import fits
from snowball_run_pipeline import detector1_with_snowball_correction
from wispsub import wispsub
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

@dataclass   
class pipeline():
    '''
    JWST nircam image pipeline 
    author: Zhongyi Zhang
        - Almostly bases on Bagley's ceers nircam image reduction pipeline.
        - elaborate discussion can be found here: https://doi.org/10.3847/2041-8213/acbb08
    -----------------------------------------------------------------
    Initial parameters:
    ------------------------
    stage1_dir: str
        directory where store pipeline stage1's intermediate results.
        *ramp.fits
        *trapsfilled.fits
        *rate_prewisp.fits(some)
        *rate_segm_cold.fits(some)
        *wisp.pdf(some)
        *rateints.fits
        *rate_pre1f.fits 
        *rate_1fmask.fits
        *rate_horiz.fits
        *rate_vert.fits
        *rate.fits

    stage2_dir: str
        directory where store pipeline stage2's intermediate results.
        *cal.fits

    stage3_dir: str
        directory where store pipeline stage3's single exposure's intermediate results.
        *tweakreg.fits
        *skymatch.fits
        *a3001_bkgsub.fits
        *a3001_crf.fits
        *a3001_match.fits

    mosaic_dir: str
        directory where store pipeline stage3's  mosaic's intermediate results.
        *mosaic_resample.fits
        *mosaic_bkg_sub.fis
    '''
    stage1_dir = "."
    stage2_dir = "."
    stage3_dir = "."
    mosaic_dir = "."

    def stage1_wf(self, uncalfile , stage1_and_snowball = True, wisp = True, striping = True, mask_threshold = 0.8):
        '''
        Nircam image reduction stage1
        Do jwst Detector1Pipeline and some artifacts: (snowball, wisp and 1/f noise(striping pattern) correction.
        ---------------------------------------------------
        parameters:
        ------------
        uncalfile: str
            address of a uncal.fits to be processed. 
            like: /mnt/data/CEERS/NIRCAM/uncals/jw01345001001_02201_00001_nrca1_uncal.fits
            not a directory, so for multiple images, please loop the dir when cite this function.
        stage1_and_snowball: boolean
            whether to run this step.
        wisp: boolean
            whether to run this step.
        striping: boolean 
            whether to run this step.
        '''
        uncalbase = os.path.basename(uncalfile)
        dataset = uncalbase.split("_uncal.fits")[0] #file's basic name
        inputdir = uncalfile.split(uncalbase)[0] #Nonsence, just for syntax
        #run stage1 and snowball
        if stage1_and_snowball:
            detector1_with_snowball_correction(dataset,inputdir,output_dir = self.stage1_dir, maxcores = "half")
            #dataset_0_*.fits is rate file
            #dataset_1_*.fits is rateints file

            #change the name.
            rampfit = os.path.join(self.stage1_dir, "%s_0_rampfitstep.fits"%dataset)
            rampintsfit = os.path.join(self.stage1_dir, "%s_1_rampfitstep.fits"%dataset)
            rate = os.path.join(self.stage1_dir, "%s_rate.fits"%dataset)
            rateints = os.path.join(self.stage1_dir, "%s_rateints.fits"%dataset)
            os.rename(rampfit, rate)        
            os.rename(rampintsfit, rateints)

            # change the fill_val from nan to 0.
            with fits.open(rate, mode = "update") as hdul:   #update is important! 
                sci = hdul[1].data
                sci[np.where(np.isnan(sci))] = 0
                hdul["SCI"].data = sci
                hdul.flush()


        #run wisp subtraction
        #⚠️调整WISPDIR: Why the wisp template version is 2 while jwst version and crds version is the latest?
        # Assuming wisp artifact only exsit in a3, a4 and b3, b4 detector at F150W and F200W band , so for most exposure don't execute this step.
        # It will judge automatically.
        if wisp:
            ratefile = os.path.join(self.stage1_dir, "%s_rate.fits"%dataset)
            origfilename = ratefile.replace(".fits","_prewisp.fits") ##原本rate.rate_prewisp.fits
            wisp = wispsub(self.stage1_dir, self.stage1_dir)
            wisp.fit_wisp_feature(ratefile, origfilename, fit_scaling = True)

        # #run 1overf noise subtraction
        if striping: 
            stripfile = os.path.join(self.stage1_dir, "%s_rate.fits"%dataset) 
            pre1f = stripfile.replace('rate.fits', 'rate_pre1f.fits') 
            striping = striping_noise(self.stage1_dir, self.stage1_dir, MASKTHRESH = mask_threshold) 
            striping.measure_striping(stripfile, pre1f, save_patterns = True) 


    def stage2_ff(self,ratefile):
        '''
        Do jwst Image2Pipeline in default configuration.    
        Correction to WCS, flat_fielding, flux calibration, etc
        ---------------------------------------------------
        input: *rate.fits
        output: *_cal.fits 
        '''
      
        Image2Pipeline.call(ratefile,output_dir = self.stage2_dir ,steps = {'bkg_subtract':{'skip':False}, 'resample':{'skip':True}}) 


    def asn_creation(self,in_suffix, out_suffix , input_dir = "None", output_dir = "None", type:{"single","multiple"} = "single"):
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
        Do JWST Image3Pipeline's single exposure part, including assign_mtwcs, tweakreg, sky_match, outlier_detection.
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
            asn_dir = self.mosaic_dir
    
        if update_wcs:
            #generate association file for tweakreg step.
            fil = self.asn_creation(in_suffix = "cal", out_suffix = "tweakreg",input_dir=input_dir, output_dir=asn_dir)
            json_t = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "tweakreg"))

            TweakRegStep.call(json_t, use_custom_catalogs = use_custom_catalogs,catfile = rel_catfile,  catalog_format = "ecsv",
                                        brightest = 400, npixels = 20, contrast = 0.005, nclip = 5, sigma = 3, 
                                        abs_refcat = abs_refcat, abs_fitgeometry = "general", save_results = True, 
                                        output_dir = self.stage3_dir, suffix = "tweakreg"
                                    )
        #do sky_match(temporarily can't subtract the bkg) and outlier detection 
        if skymatch:
            
            fil = self.asn_creation(in_suffix = "tweakreg", out_suffix = "sky_match", output_dir=asn_dir)
            json_s = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "sky_match"))
            SkyMatchStep.call(json_s,output_dir = self.stage3_dir, suffix = "skymatch", save_results = True, skymethod = "local", 
                              output_use_index = False, output_use_model = True)
        if outlier_detection:

            fil = self.asn_creation(in_suffix = "skymatch", out_suffix = "outlier", output_dir=asn_dir)
            json_o = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "outlier"))
            OutlierDetectionStep.call(json_o, output_dir = self.stage3_dir, suffix = "crf",save_results = True,
                                       output_use_index = False, output_use_model = True)

            #rename the output file 
            crfs = sorted(glob("{0}/*_crf.fits".format(self.stage3_dir)))   
            for crf in crfs: 
                p = Path(crf) 
                p.rename(p.with_name(p.stem.replace("skymatch_a3001_crf", "_a3001_crf") + p.suffix)) 
 
        #correct for the variance map(for 0 value in bad area which don't contribute varicance) and sky_match(for small overlapping area)
        if sky_wcs_var:
            crfs = sorted(glob("{0}/*_crf.fits".format(self.stage3_dir)))  
            for crf in crfs:
                swv = sky_wcs_var_class()
                swv.INPUTDIR = self.stage3_dir
                swv.OUTPUTDIR = self.stage3_dir
                swv.MASKDIR = self.stage1_dir
                swv.process(os.path.basename(crf))
            #By default, get _a3001_match.fits, which is ready to do mosaic. and _a3001_bkgsub_1.fits.

    def stage3_part2(self, asn_dir = "None", make_mosaic = True, final_bkgsub = True):
        '''
        Do mosaic creation (resample step) and final background subtraction.
        '''

        fil = self.asn_creation(in_suffix = "a3001_match", out_suffix = "mosaic", input_dir = self.stage3_dir,output_dir = asn_dir)

        if make_mosaic:
            json_m = os.path.join(asn_dir, "nircam_{0}_{1}.json".format(fil, "mosaic"))
            mosaic = ResampleStep.call(json_m, crpix = [26678.5,-724.5], crval = [214.825,52.825],rotation = -49.7,
                                    output_dir = self.mosaic_dir, save_results = True, 
                                    pixfrac =1.0, pixel_scale = 0.03,output_shape = [10600,4800],
                                    fillval = "NAN")
        #get nircam_{filter}_mosaic_resample.fits file 

        if final_bkgsub:
            resample = "{0}/nircam_{1}_mosaic_resample.fits".format(self.mosaic_dir, fil)
            suffix = "bkg_sub"
            background_and_tiermask(resample, suffix, resample.split("/nircam")[0], self.mosaic_dir)