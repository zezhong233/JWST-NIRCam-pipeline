from astropy import stats as astrostats
from astropy.convolution import (convolve_fft,Gaussian2DKernel)
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint
from astropy.stats import mad_std
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

#供ProcessPoolExecutor使用的顶层函数
def _worker_obtain_stats(index,sci,apersize, mask, max_counts,height, width,exclude_percential):
    flux_sums = []   #记录一个aperture内的flux总和
    medians = []     #记录一个aperture内的median
    count = 0        

    while count < max_counts:    
        #生成随机数
        x = np.random.randint(apersize, height - apersize)
        y = np.random.randint(apersize, width - apersize )
        #找到圈里面的像素位置   
        x_coords, y_coords = np.meshgrid(
                np.arange(x-apersize,x+apersize+1),
                np.arange(y-apersize,y+apersize+1),
                indexing = "ij")# x_coords是一个数组对应位置存储着原sci图像坐标索引值,y_coords同理
        distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        valid_1 = distances <= apersize #判断是不是在圈里   
        x_coords = x_coords[valid_1] 
        y_coords = y_coords[valid_1] #找出圈内的像素位置和value 这里的coords是一个一维数组
        #判断这个随机数是否有效
        region_mask = mask[x_coords,y_coords]
        radio = np.count_nonzero(region_mask) / region_mask.size
        if radio <= exclude_percential: 
            count += 1 
            A = len(x_coords) 
            pixels = sci[x_coords, y_coords]
            reigon_mask = mask[x_coords, y_coords]
            good_pixels = pixels[~reigon_mask]

            flux_sums.append(np.sum(good_pixels) * (pixels.size/good_pixels.size))
            medians.append(np.median(good_pixels))
    return index,A,flux_sums,medians

@dataclass
class depth_measure():
    tier_nsigma: list = (1.5,1.5,1.5,1.5)
    tier_npixels : list = (15,5,3,3)
    tier_kernel_size: list = (25, 15, 5, 2)
    tier_dilate_size: list = (33, 25, 21, 19)
    ring_radius_in:  float = 80
    ring_width: float = 4
    max_counts:int = 2000
    exclude_percential = 0
    r_min:int = 2
    r_max:int = 50
    step:int = 3

    def __init__(self, sci, pixel_scale = 0.03, PIXAR_SR = 1):
        self.sci = sci
        self.PIXAR_SR = PIXAR_SR
        self.pixel_scale = pixel_scale
        self.width = sci.shape[1]
        self.height = sci.shape[0]

#生成mask
    def _tier_mask(self, mask, tiernum = 0):
        #换掉mask的地方
        background_rms = astrostats.biweight_scale(self.sci[~mask])
        background_level = astrostats.biweight_location(self.sci[~mask]) 
        replaced_img = np.choose(mask,(self.sci, background_level))
        #fft卷积
        convolved_difference = convolve_fft(replaced_img,Gaussian2DKernel(self.tier_kernel_size[tiernum]),allow_huge=True)
        #探测sigmentation_map 并且dilate
        seg_detect = detect_sources(convolved_difference, threshold=self.tier_nsigma[tiernum] * background_rms , npixels=self.tier_npixels[tiernum], mask=mask)
        if seg_detect is None:
            print(f"No sources detected in tier #{tiernum}")
            return mask
        if self.tier_dilate_size[tiernum] == 0:
            mask = seg_detect.make_source_mask()      
        else:
            footprint = circular_footprint(radius=self.tier_dilate_size[tiernum])
            mask = seg_detect.make_source_mask(footprint = footprint)
        return mask      

    def _mask_sources(self, origin_mask): 
        current_mask = origin_mask != 0
        for tiernum in tqdm(range(len(self.tier_nsigma)),desc = "Generating Mask",leave = True):
            mask = self._tier_mask(current_mask, tiernum=tiernum)
            current_mask = np.logical_or(current_mask, mask)
        return current_mask
    
    def generate_mask(self, off_detector = None):
        coverage_mask = np.zeros_like(self.sci,bool)
        if off_detector is not None:
            coverage_mask[off_detector == 1] = 1
        mask = self._mask_sources(coverage_mask)  
        print("mask_size:",mask.size)
        return mask

    def _cal_zp(self):
        zp_AB = -6.10 - 2.5 * np.log10(self.PIXAR_SR)
        return zp_AB

    def _cal_stats(self, mask):
        apersize_array = np.arange(self.r_min,self.r_max,self.step)
        # apersize_array = np.logspace(np.log10(self.r_min), np.log10(self.r_max), 30).astype(int)
        print(len(apersize_array))
        As = []
        median_stds = []
        nmads = []
        n = len(apersize_array)
        tasks = [(index,self.sci,apersize,mask, self.max_counts, self.height, self.width, self.exclude_percential) for index, apersize in enumerate(apersize_array)]   

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker_obtain_stats, *t) for t in tasks}
            for future in tqdm(as_completed(futures), total = n,desc="计算进度"):  
                index,A,flux_sums,medians = future.result()
                As.append(A),nmads.append(mad_std(flux_sums)),median_stds.append(np.std(medians))

        return As, nmads, median_stds
    
    def measure_aperture_sigma(self, mask, Return = True):   
        As, nmads, _ = self._cal_stats(mask)
        self.As = As
        self.nmads = nmads
        if Return:
            return As, nmads
        
    def measure_surface_sigma(self, mask):
        As, _, median_stds = self._cal_stats(mask)
        self.As = As
        self.median_stds = median_stds
        return As, median_stds
    
    def measure_single_sigma (self,mask, Return = True):
        As = self.As
        pixel_values = []
        pixel_stds = []
        count = 0
        while count <= self.max_counts:
            x = np.random.randint(0, self.height)
            y = np.random.randint(0, self.width)
            valid = mask[x,y] == 0  
            if valid:
                count += 1
                pixel_values.append(self.sci[x,y])
        single_std = np.std(pixel_values)
        for A_specify in As:
            pixel_stds.append(np.sqrt(A_specify) * single_std)
        self.pixel_stds = pixel_stds
        self.single_std = single_std
        if Return:
            return pixel_stds,single_std
    
    def cal_image_depth(self,sigma,zp_AB):
        image_depth = -2.5 * np.log10(5 * sigma) + zp_AB #计算namd对应的星等
        return image_depth
    
    def cal_surface_depth(self, a, zp_AB):
        surbright_depth = -2.5 * np.log10(5 * a / (self.pixel_scale**2)) + zp_AB 
        return surbright_depth
    
    def _fitting(self):
        Ns = np.sqrt(self.As)
        y = self.nmads 
        lr = stats.linregress(np.log(Ns),np.log(y))
        beta = lr.slope
        intercept = math.exp(lr.intercept)
        return beta, intercept
    
    def plot_apersigma(self, save = False, text = "result", plot_single = False):

        fig,ax1 = plt.subplots(figsize = (7,7)) 
        Ns = np.sqrt(self.As)
        if plot_single:
            ax1.scatter(Ns,self.pixel_stds,label = "$\sigma_{single}$") 
        ax1.scatter(Ns, self.nmads,label = "nmads")
        ax1.set_xlabel("Ns",fontsize = 24)
        ax1.set_ylabel("$\sigma$",fontsize = 24)
        beta, coeff = self._fitting()
        print(f"The coefficiency is {coeff}")
        print(f"The beta is {beta}")
        x_min = np.sqrt(self.As)[0]
        x_max = np.sqrt(self.As)[-1]
        x_smooth = np.linspace(x_min, x_max, 1000)
        y_smooth =  coeff * (x_smooth ** beta)
        ax1.plot(x_smooth,y_smooth, label = f"beta = {beta:.3f}")
        ax1.legend(fontsize = 18)
        ax1.set_ylim(0, 2)
        if save:
            fig = ax1.get_figure()
            fig.savefig(f'{text}.png', dpi=300, bbox_inches='tight')

        # ax1.set_xscale('log')
        # ax1.set_yscale('log')


    def plot_surface_sigma(self):

        fig,ax2 = plt.subplots(figsize = (7,7))
        Ns = np.sqrt(self.As)
        ax2.scatter(Ns, self.median_stds,s = 3,label = "sigma_median")
        ax2.set_xlabel("Ns",fontsize = 24)
        ax2.set_ylabel("$\sigma_{median}$",fontsize = 24)

        ax2.axhline(y = self.median_stds[-1],color = "r", linestyle = "--",label = f"$\sigma$ = {self.median_stds[-1]:.5e}")
        ax2.legend(fontsize = 24)