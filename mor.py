import numpy as np
import os
import re
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.cm as mcm
import matplotlib.colors as mcol
from matplotlib.patches import Ellipse
import numpy.ma as nma
import petrofit as ptf
from astropy.io import fits
from astropy.visualization import ManualInterval,LogStretch
from astropy.cosmology import Planck18 as cosmo
import time

from photutils.psf import MoffatPSF
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources, SourceFinder, SourceCatalog
from photutils.utils import circular_footprint
from astropy.convolution import convolve

import statmorph
from data_get import *
from make_rgb import *

plt.style.use('./presentation.mplstyle')

channels = {'g':mcm.cividis,'r':mcm.viridis,'i':mcm.inferno,'z':mcm.Greys_r}
rd2deg = (180*3600)/np.pi
pixsc = 0.262

def rgb_cutout(img_data):
    imax = 0
    for img in img_data[::-1]:
        val = np.percentile(img,99.5)
        if val > imax: imax = val
    
    clip_intvls = 3*[ManualInterval(vmin=0, vmax=imax)]
    clip_intvls[2] = ManualInterval(vmin=0, vmax=2*imax)
    
    return make_rgb(img_data[2],img_data[1],img_data[0],interval=clip_intvls,stretch=LogStretch(a=5))

def image_bkg(img_dat,c=['g','r','i','z']):
    img_sub = np.zeros((4,128,128))
    img_rms = np.zeros((4,128,128))
    glob_rms = np.zeros(4)

    img_arr = img_dat[0].data
    img_hdr = img_dat[0].header
    img_invvar = img_dat[1].data

    w = wcs.WCS(img_hdr,naxis=2)

    flags=[False]*4 

    #fig,ax=plt.subplots(2,len(c),figsize=(6*len(c),14),sharey=True)
   
    for i,key in enumerate(channels):        
        threshold = detect_threshold(img_arr[i,:,:], nsigma=1.0)
        segment_img = detect_sources(img_arr[i,:,:], threshold, npixels=10,connectivity=8)

        try: mask = segment_img.make_source_mask(size=2)
        except: 
            flags[i]=True
            continue
        
        var = np.reciprocal(img_invvar[i,:,:])
        bkg = Background2D(img_arr[i,:,:].astype(img_arr[i,:,:].dtype.newbyteorder('=')), ( 32, 32), filter_size=(5,5), mask=mask)

        if np.mean(var)>100 or np.any(var<0): flags[i]=True
        print(key+' Background:',bkg.background_median,key+' RMS^2:',np.square(bkg.background_rms_median),key+' Var:',np.mean(var))

        img_sub[i] = img_arr[i,:,:]-np.abs(bkg.background)
        img_rms[i] = np.sqrt(var)
        glob_rms[i] = bkg.background_median

    #plt.subplots_adjust(wspace=0.0,hspace=0.05)
    #plt.show()

    return img_sub, img_hdr, img_rms, glob_rms, flags

def gaussian_2d(x, y, x0=0.0, y0=0.0, sigma=1.0): return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma**2))

def psf_grid(d_psf,fwhm, psf_type='gaussian'):
    x = np.arange(-d_psf, d_psf+1)
    y = np.arange(-d_psf, d_psf+1)
    x_grid, y_grid = np.meshgrid(x, y)
    
    if psf_type=='gaussian':
            psf = gaussian_2d(x_grid, y_grid, sigma=fwhm)
            return psf/np.sum(psf)
    elif psf_type=='moffat':
            model = MoffatPSF(x_0=0.0, y_0=0.0, alpha=fwhm, beta=2.0)
            psf = model(x_grid, y_grid)
            return psf/np.sum(psf)

def morph_main(sources):
    model_psf = psf_grid(d_psf=8,fwhm=3)

    morph_par,petro_par,sersic_par = [],[],[]
    imzer = np.zeros((128,128))
    xx, yy = np.meshgrid(np.arange(0,128,1),np.arange(0,128,1))
    
    t1 = time.time()
    
    for ii,key in sources.iterrows():
            if ii<800 or ii>850: continue
            pos = np.array([key['ra'],key['dec']])
            print(ii,pos,key['zspec'],key['logM'])
            pos_coord = SkyCoord(ra=pos[0]*u.degree, dec=pos[1]*u.degree, frame='icrs')
            
            download_cutout(pos.reshape(1,2))
            filename = 'cutout_{:.4f}_{:.4f}.fits'.format(pos[0],pos[1])
            #print(filename)
            try: data = fits.open("cutouts/"+filename)
            except: continue
    
            img_data,img_hdr,img_rms,glob_rms,flags = image_bkg(data)
            if (flags[0] or flags[1] or flags[3]): continue
    
            w = wcs.WCS(img_hdr,naxis=2)
            wc = [int(i) for i in w.world_to_pixel(pos_coord)]
    
            segmaps_lo,segmaps_hi = np.zeros((4,128,128)),np.zeros((4,128,128))
       
            for jj in range(4):
                    if flags[jj]==True: continue
                    #image_plot(ax[jj],img_data[jj],cmp=mcm.Greys)
                    convolved_data = convolve(img_data[jj], model_psf)
    
                    finder = SourceFinder(npixels=10, nlevels=32, contrast=0.0005, progress_bar=False)
                    segment_map = finder(convolved_data, img_rms[jj])
    
                    if segment_map==None: 
                        flags[jj]==True
                        continue
            
                    cat = SourceCatalog(img_data[jj], segment_map, convolved_data=convolved_data)
                    tbl = cat.to_table()                    
                        
                    target = np.argmin(np.sqrt((tbl['xcentroid']-wc[0])**2+(tbl['ycentroid']-wc[1])**2))
                    segmaps_lo[jj] = (1*(segment_map==(target+1)))
                    segmaps_hi[jj] = (1*((segment_map!=imzer)&(segment_map!=(target+1))))
    
    
            if (flags[0] or flags[1]): continue
    
            comp_segmap = 1* (np.logical_and(segmaps_lo[0],segmaps_lo[1])&(np.logical_xor(segmaps_lo[2],flags[2])|np.logical_xor(segmaps_lo[3],flags[3])))
            comp_mask = 1* np.logical_or(np.logical_or(segmaps_hi[0],segmaps_hi[1]),np.logical_or(segmaps_hi[2],segmaps_hi[3]))
    
            if comp_segmap[63,63]<1: continue
            
            fig,ax=plt.subplots(figsize=(16,8))
    
            rgb_img = rgb_cutout(img_data)
            #ax.imshow(comp_segmap, origin='lower')
            #ax[0].scatter(wc[0], wc[1],marker='*',color='red',s=40,zorder=101)
            
            morph_arr,petro_arr,sersic_arr = np.zeros((4,12)),np.zeros((4,6)),np.zeros((4,5))
    
            img_copy = [img_data[k].copy() for k in range(3)]
            ax.imshow(rgb_cutout(img_copy), origin='lower',extent=[pos[0]-16.506,pos[0]+16.506, pos[1]-16.506, pos[1]+16.506])
    
            for k in range(3): np.putmask(img_copy[k],~(comp_segmap.astype('bool')),0)
    
            ax.imshow(rgb_cutout(img_copy), origin='lower',extent=[pos[0]-16.506,pos[0]+16.506, pos[1]-16.506, pos[1]+16.506],alpha=0.5)
            #ax.scatter(pos[0], pos[1],marker='*',color='red',s=40,zorder=101)
            for jj,k in enumerate(channels):
                if flags[jj]==True: continue
                obs_psf = psf_grid(d_psf=16,fwhm=key['psfsize_'+k]/pixsc,psf_type='moffat')
                morph = statmorph.source_morphology(img_data[jj], comp_segmap, weightmap=img_rms[jj],mask=comp_mask.astype('bool'),psf=obs_psf)[0]
    
                morph_arr[jj,:-1] = ii,morph.gini,morph.m20,morph.gini_m20_bulge,morph.gini_m20_merger,morph.concentration,morph.asymmetry,morph.smoothness,morph.shape_asymmetry,morph.flag,morph.sn_per_pixel
                petro_arr[jj,:] = ii, morph.xc_asymmetry,morph.yc_asymmetry,morph.rpetro_ellip,morph.orientation_asymmetry,morph.ellipticity_asymmetry
    
                x,y,th = xx-petro_arr[jj,0], yy-petro_arr[jj,1],petro_arr[jj,3] 
                zz = np.sqrt(np.square((x*np.cos(th)+y*np.sin(th))/petro_arr[jj,2]) + np.square((x*np.sin(th)-y*np.cos(th))/(petro_arr[jj,2]*(1-petro_arr[jj,4]))))
                morph_arr[jj,-1] = np.sum(1*((comp_segmap>0)))/np.sum(1*(zz<1))
    
                sersic_arr[jj,:] = ii,morph.sersic_n,morph.sersic_rhalf,morph.sersic_ellip,morph.flag_sersic
    
            pos_asymm = w.pixel_to_world_values(petro_arr[1,0], petro_arr[1,1])
            ax.scatter(pos_asymm[0], pos_asymm[1],marker='*',color='red',s=40,zorder=101)
    
            
            e = Ellipse(xy=[pos_asymm[0], pos_asymm[1]],width=2*petro_arr[1,2]*pixsc, height=2*petro_arr[1,2]*(1-petro_arr[1,4])*pixsc,
                    angle=petro_arr[1,3] * 180/np.pi,lw=2, fill=False,color='azure',ls='--')
            ax.add_artist(e)
            
            ax.text(pos[0]-16.5,pos[1]-14.5,r'${{\rm log}}M_{{\ast}}={0:.2f}$'.format(key['logM']),fontsize=30,color='white')
            ax.text(pos[0]+2.5,pos[1]-14.5,r'$z ={0:.2f}$'.format(key['zspec']),fontsize=30,color='white')
            ax.text(pos[0]+2.5,pos[1]+12.5,r'${{\it g-r}}={0:.2f}$'.format(key['g-r']),fontsize=30,color='white')
    
            ax.set_xlabel(r'$RA \ (deg)$',fontsize=40)
            ax.set_ylabel(r'$DEC \ (deg)$',fontsize=40)
    
            plt.savefig('post_cutouts/galseg_{:.4f}_{:.4f}.png'.format(pos[0],pos[1]),bbox_inches='tight')
            #plt.show()
            
            morph_par.append(morph_arr)
            petro_par.append(petro_arr)
            sersic_par.append(sersic_arr)
    
    t2= time.time()
    print(t2-t1)

    morph_all = np.array(morph_par)
    print(np.shape(morph_all))

    np.save("morph_array.npy", morph_all)

    sersic_all = np.array(sersic_par)
    print(np.shape(sersic_all))

    np.save("sersic_array.npy", sersic_all)

    petro_all = np.array(petro_par)
    print(np.shape(petro_all))

    np.save("petro_array.npy", petro_all)

    return morph_all, sersic_all, petro_all

if __name__ == '__main__':
    v1saga = pd.read_csv('v1saga_dwarfs.csv')
    print(np.shape(v1saga))

    mrph, srsc, petr = morph_main(v1saga)

    
