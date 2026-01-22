import os
import sep
import re
import multiprocessing
import time
from functools import partial
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy.ma as nma
import fitsio
from astropy.io import fits
import statmorph
from data_get import *
from make_rgb import *


channels = {'g':mcm.cividis,'r':mcm.viridis,'i':mcm.inferno,'z':mcm.Greys_r}

def image_plot(ax,img,cmp):
    m, s = np.mean(img), np.std(img)
    ax.imshow(img, interpolation='nearest', cmap=cmp, vmin=m-s, vmax=m+s, origin='lower')

def rgb_cutout(img_data):
    imax = 0
    for img in img_data[::-1]:
        val = np.percentile(img,99.5)
        if val > imax: imax = val
    
    clip_intvls = 3*[ManualInterval(vmin=0, vmax=imax)]
    clip_intvls[2] = ManualInterval(vmin=0, vmax=100.)
    
    return make_rgb(img_data[2],img_data[1],img_data[0],interval=clip_intvls,stretch=LogStretch(a=50))
    

def image_bkg(img_dat,c=['g','r','i','z']):
    img_sub = []
    img_rms = []
    glob_rms = []

    img_arr = img_dat.data
    img_hdr = img_dat.header

    #w = wcs.WCS(img_hdr,naxis=2)
    #world_coords = w.all_pix2world([[0],[255]], 1)[0]
 
    for i,key in enumerate(channels):

        bkg = sep.Background(img_arr[i,:,:].byteswap().newbyteorder())
        #print(key+' Background:',bkg.globalback,key+' RMS:',bkg.globalrms)
        glob_rms.append(bkg.globalrms)
                
        bkg_image,bkg_rms = bkg.back(),bkg.rms()
        img_rms.append(bkg_rms)
        
        img_sub.append(img_arr[i,:,:] - bkg_image)

    return np.stack(img_sub,0),img_hdr, np.stack(img_rms,0),glob_rms

def gaussian_2d(x, y, x0=0.0, y0=0.0, sigma=1.0): return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma**2))

def create_gaussian_2d_grid(d_psf,fwhm, **kwargs):
    x = np.arange(-d_psf, d_psf+1)
    y = np.arange(-d_psf, d_psf+1)
    x_grid, y_grid = np.meshgrid(x, y)
    
    return gaussian_2d(x_grid, y_grid, sigma=fwhm)

model_psf =create_gaussian_2d_grid(d_psf=6,fwhm=3)
model_psf /= np.sum(model_psf)


def galaxy_seg_morph(ii,key):
        pos = np.array([key['RA'],key['DEC']])
        print(pos)
        pos_coord = SkyCoord(ra=pos[0]*u.degree, dec=pos[1]*u.degree, frame='icrs')
        
        download_cutout(pos.reshape(1,2))

        filename = 'cutout_{:.4f}_{:.4f}.fits'.format(pos[0],pos[1])
        print(filename)
    
        try: 
            data = fits.open("cutouts/"+filename)
            
            img_data,img_hdr,img_rms,glob_rms = image_bkg(data[0])
            
            if np.any(img_data[2]>0): 
            
                #fig,ax=plt.subplots(1,4,figsize=(24,8),sharey=True)
                
                w = wcs.WCS(img_hdr,naxis=2)
                wc = [int(i) for i in w.world_to_pixel(pos_coord)]
                
                segmaps_hi = []
                for jj,key in enumerate(channels):
                        
                        objects,seg_map = sep.extract(img_data[jj],4,err=img_rms[jj],
                                                      segmentation_map=True,deblend_cont=0.001,
                                                      filter_type='matched',filter_kernel=model_psf)
                        target = np.argmin(np.sqrt((objects['x']-wc[0])**2+(objects['y']-wc[1])**2))
                        hi_seg = 1*(seg_map==(target+1))
                        #print(hi_seg)
                
                        segmaps_hi.append(hi_seg)
                
                comp_segmap = (segmaps_hi[0]&segmaps_hi[1])*(segmaps_hi[2]*segmaps_hi[3])
                
                if comp_segmap[63,63]>0:
                    #fig,ax=plt.subplots(1,2,figsize=(16,8),sharey=True)
                
                    rgb_img = rgb_cutout(img_data)
                    #ax[0].imshow(rgb_img, origin='lower')
                    #ax.imshow(comp_segmap, origin='lower')
                    #ax[0].scatter(wc[0], wc[1],marker='*',color='red',s=40,zorder=101)
                
                    for k in range(3): np.putmask(img_data[k],~(comp_segmap.astype('bool')),0)
                    rgb_img = rgb_cutout(img_data)
                
                    #ax[1].imshow(rgb_img, origin='lower')
                    #ax.imshow(comp_segmap, origin='lower')
                    #ax[1].scatter(wc[0], wc[1],marker='*',color='red',s=40,zorder=101)
                
                    #plt.show()
                    morph_arr = np.zeros((4,9))
                    for jj,key in enumerate(channels):
                        morph = statmorph.source_morphology(img_data[jj], comp_segmap, weightmap=img_rms[jj])[0]
                        morph_arr[jj,:] = ii,morph.gini,morph.m20,morph.gini_m20_bulge,morph.gini_m20_merger,morph.concentration,morph.asymmetry,morph.smoothness,morph.flag
                
                return morph_arr


def process_chunk(sources_chunk, process_func):
    """Process a chunk of items using the provided function."""
    morph_all = []

    for ii,key in sources_chunk.iterrows():
        galaxy_seg_morph(ii,key)
        morph_all.append(morph_arr)
    return morph_all


def split_processing(sources, num_processes=None, process_func=galaxy_seg_morph):
    """Split processing across multiple processes and consolidate results.
    
    Args:
        items: The list of items to process
        num_processes: Number of processes to use (defaults to CPU count)
        process_func: The function to apply to each item
        
    Returns:
        List of processed items
    """
    # Use CPU count if num_processes not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print("Processes:",num_processes)
    # Ensure we don't create more processes than items
    #num_processes = min(num_processes, sources.size())
    
    # Split the items into roughly equal chunks
    chunk_size = sources.size() // num_processes
    chunks = [sources.iloc[i:i + chunk_size,:] for i in range(0, len(items), chunk_size)]
    
    # Adjust the chunks if we ended up with more chunks than processes
    while len(chunks) > num_processes:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    
    # Create a process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map chunks to processes
        chunk_processor = partial(process_chunk, process_func=process_func)
        chunk_results = pool.map(chunk_processor, chunks)
    
    # Consolidate results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return results

if __name__ == "__main__":
    # Example usage
    start_time = time.time()

    sources = pd.read_csv('v1saga_dwarfs.csv',nrows=100)
    print(np.shape(sources))
    #download_cutout(sources)
    
    # Process the items across multiple processes
    results = split_processing(sources, num_processes=4,  # Use 4 processes
        process_func=galaxy_seg_morph)
    
    end_time = time.time()
    
    print(f"Processed {len(items)} items in {end_time - start_time:.2f} seconds")
    print(f"First few results: {results[:5]}")
    
    # Verify the results
    expected = [item * 3 for item in items]
    assert results == expected, "Results don't match expected output!"
    print("All results verified correctly!")