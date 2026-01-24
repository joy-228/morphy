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
from matplotlib.patches import Ellipse
import numpy.ma as nma
#import petrofit as ptf
from astropy.io import fits
from astropy.visualization import ManualInterval,LogStretch
from astropy.cosmology import Planck18 as cosmo
import time

from photutils.psf import MoffatPSF
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources, SourceFinder, SourceCatalog
from astropy.convolution import convolve

import statmorph
from make_rgb import *

plt.style.use('./presentation.mplstyle')

bands = {'g':'tab:blue','r':'tab:green','i':'tab:orange','z':'tab:red'}
rd2deg = (180*3600)/np.pi
pixsc = 0.262
npix = 128

def read_sources(filename,Nsamp=0):
    data = np.genfromtxt(filename, delimiter=',',skip_header=True)

    if Nsamp>0: data = data[npr.choice(len(data), Nsamp)]
    return data

def request_apply(url,filename,out_dir):
    if not os.path.exists(out_dir+filename):
                print(url)
                print('downloading -- ', filename) 

                response = requests.get(url)
                
                if response.status_code == 200:
                    with open(out_dir+filename, 'wb') as file:
                        file.write(response.content)


def download_cutout(sample,  filters='griz',size=npix,out_dir='cutouts/'):
    # will download the files from NERSC into the work_dir folder

    for j in range(len(sample)):
        ra,dec = sample[j,0],sample[j,1]
        filename = 'cutout_{:.4f}_{:.4f}.fits'.format(ra,dec)
        url = ('https://www.legacysurvey.org/viewer/fits-cutout?ra={:.4f}&dec={:.4f}&size='+str(size)+'&layer=ls-dr10-south&invvar&pixscale=0.262&bands='+filters).format(ra,dec)

        request_apply(url,filename,out_dir) 

def rgb_cutout(img_dat):
    imax = 0
    for img in img_dat[::-1]:
        val = np.percentile(img,99.5)
        if val > imax: imax = val
    
    clip_intvls = 3*[ManualInterval(vmin=0, vmax=imax)]
    clip_intvls[2] = ManualInterval(vmin=0, vmax=2*imax)
    
    return make_rgb(img_dat[2],img_dat[1],img_dat[0],interval=clip_intvls,stretch=LogStretch(a=5))

def image_bkg(img_dat):
    img_sub,img_rms = np.zeros((4,npix,npix)),np.zeros((4,npix,npix))
    glob_rms = np.zeros(4)

    img_arr,img_hdr,img_invvar = img_dat[0].data,img_dat[0].header,img_dat[1].data

    w = wcs.WCS(img_hdr,naxis=2)

    flags=[False]*4 
   
    for i,key in enumerate(bands):        
        threshold = detect_threshold(img_arr[i,:,:], nsigma=1.0)
        segment_img = detect_sources(img_arr[i,:,:], threshold, npixels=10,connectivity=8)

        try: mask = segment_img.make_source_mask(size=2)
        except: 
            flags[i]=True
            continue
        
        var = np.reciprocal(img_invvar[i,:,:])
        bkg = Background2D(img_arr[i,:,:].astype(img_arr[i,:,:].dtype.newbyteorder('=')), ( 32, 32), filter_size=(5,5), mask=mask)

        if np.mean(var)>100 or np.any(var<0): flags[i]=True
        #print(key+' Background:',bkg.background_median,key+' RMS^2:',np.square(bkg.background_rms_median),key+' Var:',np.mean(var))

        img_sub[i] = img_arr[i,:,:]-np.abs(bkg.background)
        img_rms[i] = np.sqrt(var)
        glob_rms[i] = bkg.background_median


    return img_sub, img_hdr, img_rms, glob_rms, flags

def image_segmap(img_dat,img_rms,psf_mod,wc,flags):
        segmaps_lo,segmaps_hi = np.zeros((4,npix,npix)),np.zeros((4,npix,npix))
        imzer = np.zeros((npix,npix))
   
        for ii,key in enumerate(bands):        
                if flags[ii]==True: continue
                convolved_data = convolve(img_dat[ii], psf_mod)

                finder = SourceFinder(npixels=10, nlevels=32, contrast=0.0005, progress_bar=False)
                segment_map = finder(convolved_data, img_rms[ii])

                if segment_map==None: 
                    flags[jj]=True
                    continue
        
                cat = SourceCatalog(img_dat[ii], segment_map, convolved_data=convolved_data)
                tbl = cat.to_table()                    
                    
                target = np.argmin(np.sqrt((tbl['xcentroid']-wc[0])**2+(tbl['ycentroid']-wc[1])**2))
                segmaps_lo[ii] = (1*(segment_map==(target+1)))
                segmaps_hi[ii] = (1*((segment_map!=imzer)&(segment_map!=(target+1))))
                '''
                flux_arr, area_arr, error_arr = ptf.source_photometry(cat[target], img_dat[jj], segment_map, r_list, error=img_rms[jj],bg_sub=False, plot=False)
                petr = ptf.Petrosian(r_list, area_arr, flux_arr, flux_err=error_arr)
                print("{:0.4f} ± {:0.4f} pix".format(petr.r_petrosian, petr.r_petrosian_err))
                '''

        if not (flags[0] or flags[1]): 
            comp_segmap = 1* (np.logical_and(segmaps_lo[0],segmaps_lo[1])&(np.logical_xor(segmaps_lo[2],flags[2])|np.logical_xor(segmaps_lo[3],flags[3])))
            comp_mask = 1* np.logical_or(np.logical_or(segmaps_hi[0],segmaps_hi[1]),np.logical_or(segmaps_hi[2],segmaps_hi[3]))
        else: flags= [True]*4
        if comp_segmap[npix//2 -1,npix//2 -1]<1: flags= [True]*4
        
        return comp_segmap,comp_mask,flags

def plot_cutout(img_dat,comp_segmap,pos,w,ptr,df_row):
        fig,ax=plt.subplots(figsize=(16,8))
        rgb_img = rgb_cutout(img_dat)
        img_copy = [img_dat[k].copy() for k in range(3)]
        ax.imshow(rgb_cutout(img_copy), origin='lower',extent=[pos[0]-16.506,pos[0]+16.506, pos[1]-16.506, pos[1]+16.506])

        for k in range(3): np.putmask(img_copy[k],~(comp_segmap.astype('bool')),0)
        ax.imshow(rgb_cutout(img_copy), origin='lower',extent=[pos[0]-16.506,pos[0]+16.506, pos[1]-16.506, pos[1]+16.506],alpha=0.5)

        #ax.scatter(pos_asymm[0], pos_asymm[1],marker='*',color='red',s=40,zorder=101)

        e = Ellipse(xy=w.pixel_to_world_values(ptr[0], ptr[1]),width=2*ptr[2]*pixsc, height=2*ptr[2]*(1-ptr[4])*pixsc,
                angle=ptr[3] * 180/np.pi,lw=2, fill=False,color='azure',ls='--')
        ax.add_artist(e)
        
        ax.text(pos[0]-16.5,pos[1]-14.5,r'${{\rm log}}M_{{\ast}}={0:.2f}$'.format(df_row['logM']),fontsize=30,color='white')
        ax.text(pos[0]+2.5,pos[1]-14.5,r'$z ={0:.2f}$'.format(df_row['zspec']),fontsize=30,color='white')
        ax.text(pos[0]+2.5,pos[1]+12.5,r'${{\it g-r}}={0:.2f}$'.format(df_row['g-r']),fontsize=30,color='white')

        ax.set_xlabel(r'$RA \ (deg)$',fontsize=40)
        ax.set_ylabel(r'$DEC \ (deg)$',fontsize=40)

        plt.savefig('post_cutouts/galseg_{:.4f}_{:.4f}.png'.format(pos[0],pos[1]),bbox_inches='tight')
        #plt.show()

def gaussian_2d(x, y, x0=0.0, y0=0.0, sigma=1.0): return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma**2))

def elliptical_transform(x,y,rp,th,e): return np.sqrt(np.square((x*np.cos(th)+y*np.sin(th))/rp) + np.square((x*np.sin(th)-y*np.cos(th))/(rp*(1-e))))

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

def morph_main(sources,inds=None):
    morph_all = []
    model_psf = psf_grid(d_psf=8,fwhm=3)
    xx, yy = np.meshgrid(np.arange(0,npix,1),np.arange(0,npix,1))
    
    t1 = time.time()
    
    for ii,key in sources.iterrows():
            if ii<1200 or ii>1300: continue
            pos = np.array([key['ra'],key['dec']])
            print(ii,'RA:',pos[0],'DEC:',pos[1],'z:',key['zspec'],'logM:',key['logM'])
            
            download_cutout(pos.reshape(1,2))
            filename = 'cutout_{:.4f}_{:.4f}.fits'.format(pos[0],pos[1])
            #print(filename)
            try: data = fits.open("cutouts/"+filename)
            except: continue
    
            img_data,img_hdr,img_rms,glob_rms,flags = image_bkg(data)
            if (flags[0] or flags[1]): continue
    
            w = wcs.WCS(img_hdr,naxis=2)
            pos_coord = SkyCoord(ra=pos[0]*u.degree, dec=pos[1]*u.degree, frame='icrs')
            wc = [int(i) for i in w.world_to_pixel(pos_coord)]
            
            comp_segmap,comp_mask,flags = image_segmap(img_data,img_rms,model_psf,wc,flags)
            if (flags[0] or flags[1]): continue
    
            morph_arr = np.zeros((4,19))
            #ax.scatter(pos[0], pos[1],marker='*',color='red',s=40,zorder=101)
            for jj,k in enumerate(bands):
                if flags[jj]==True: continue
                obs_psf = psf_grid(d_psf=16,fwhm=key['psfsize_'+k]/pixsc,psf_type='moffat')
                morph = statmorph.source_morphology(img_data[jj], comp_segmap, weightmap=img_rms[jj],mask=comp_mask.astype('bool'),psf=obs_psf)[0]
    
                morph_arr[jj,:8] = [morph.gini,morph.m20,morph.concentration,morph.asymmetry,morph.smoothness,morph.shape_asymmetry,morph.flag,morph.sn_per_pixel]
    
                morph_arr[jj,8] = morph.r20/(key['psfsize_'+k]/pixsc)
                zz = elliptical_transform(xx-morph.xc_asymmetry,yy-morph.yc_asymmetry,morph.rpetro_ellip,morph.orientation_asymmetry,morph.ellipticity_asymmetry)
                morph_arr[jj,9] = np.sum(1*((comp_segmap>0)))/np.sum(1*(zz<1))
                morph_arr[jj,10:15] = [morph.xc_asymmetry,morph.yc_asymmetry,morph.rpetro_ellip,morph.orientation_asymmetry,morph.ellipticity_asymmetry]
    
                morph_arr[jj,15:] = [morph.sersic_n,morph.sersic_rhalf,morph.sersic_ellip,morph.flag_sersic]
    
            plot_cutout(img_data,comp_segmap,pos,w,morph_arr[0,10:15],key)
    
            morph_row = [ii]
            morph_row.extend(morph_arr.ravel())
            morph_all.append(morph_row)
    
    t2= time.time()
    print(t2-t1)

    return morph_all

if __name__ == '__main__':
    v1saga = pd.read_csv('v1saga_dwarfs.csv')
    print(v1saga.shape)

    morph_all = morph_main(v1saga)

    param_names = ['gini','m20','c','a','s','a_s','flag','snr_pix','r20_psf','segm_petr','xc_as','yc_as','rpetro','orien_as','ellip_as','n','rhalf','ellip_ss','flag_ss']
    col_names = ['ind']
    for b in bands: col_names.extend([x+'_'+b for x in param_names])
    
    smorph = pd.DataFrame(morph_all, columns=col_names)
    smorph.to_csv('saga_morph.csv', index=False)
    print(smorph.shape)

    fig,ax=plt.subplots(1,5,figsize=(30,8),sharey=True)

    for i,key in enumerate(bands):
        high_snr = np.where((smorph['snr_pix_'+key]>2)&(smorph['flag_'+key]<=1))[0]
        print(len(high_snr))
    
        for j in range(5):
            ax[j].hist(smorph[param_names[j]+'_'+key][high_snr],bins=21,alpha=0.25,color=bands[key]) 
    plt.subplots_adjust(wspace=0.0)
    plt.show()
    
        
