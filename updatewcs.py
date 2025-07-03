__author__ = "Micaela B. Bagley, UT Austin"
__version__ = "0.1.0"
__license__ = "BSD3"

import os
import sys
from glob import glob
import argparse
import numpy as np

from jwst.datamodels import ImageModel
import asdf
from dataclasses import dataclass 




@dataclass 
class tweakreg_update():
    INPUTDIR = '.'
    OUTPUTDIR = '.'
    TWEAKDIR = 'tweakreg'
    FILE_SUFFIX = 'cal'
    OUTPUT_SUFFIX = 'tweakreg'

    def update_wcs(self, cal):
        """Update header with tweaked WCS saved from previous TweakReg run

        Args: 
            cal (str): filename of input cal file

        Outputs:
            - Output image with updated WCS model with suffix set by 
                OUTPUT_SUFFIX, ready for outlier detection
        """
        # open with jwst datamodels
        model = ImageModel(os.path.join(self.INPUTDIR,cal))
        
        # find tweakregged asdf file for this image
        base = cal.split('_%s.fits'%self.FILE_SUFFIX)[0]
        print('%s base: %s'%(cal, base))
        # directory of asdf wcs from tweakregged images
        tweakwcs = os.path.join(self.TWEAKDIR, '%s_tweakreg.asdf'%base)

        tweakreg = asdf.open(tweakwcs)

        wcs = tweakreg['wcs']
        wcsinfo = tweakreg['wcsinfo']

        print('%s updating wcs from %s'%(cal,tweakwcs))
        model.meta.wcs = wcs
        model.meta.wcsinfo = wcsinfo
        model.meta.cal_step.tweakreg = 'COMPLETE'

        # save output
        model.save(os.path.join(self.OUTPUTDIR, cal.replace('_%s.fits'%self.FILE_SUFFIX,
                                                    '_%s.fits'%self.OUTPUT_SUFFIX)))





