"""
Morphological Analysis Module for Galaxy Morphology Studies

This module performs morphological measurements on galaxy cutout images using
statmorph and photutils. It handles downloading cutouts from the Legacy Survey,
performing background subtraction, source segmentation, and computing various
morphological parameters (Gini, M20, concentration, asymmetry, smoothness, etc.).

Key functions:
- download_cutout: Fetch galaxy cutouts from Legacy Survey
- image_bkg: Perform background subtraction on cutout images
- image_segmap: Create segmentation maps to identify galaxy sources
- morph_main: Main pipeline for morphological analysis
"""

import numpy as np
import os
import requests
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
from astropy.visualization import ManualInterval, LogStretch
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

# Plotting color scheme for g, r, i, z bands
bands = {'g': 'tab:blue', 'r': 'tab:green', 'i': 'tab:orange', 'z': 'tab:red'}
rd2deg = (180*3600)/np.pi  # Conversion factor: radians to arcseconds to degrees
pixsc = 0.262  # Pixel scale in arcseconds/pixel (Legacy Survey)
npix = 128  # Size of cutout images in pixels

def read_sources(filename, Nsamp=0):
    """
    Read source data from a CSV file.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file containing source data
    Nsamp : int, optional
        Number of random samples to select. If 0 (default), use all data.
    
    Returns
    -------
    data : ndarray
        Array of source data from the file
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=True)

    if Nsamp > 0:
        data = data[npr.choice(len(data), Nsamp)]
    return data

def request_apply(url, filename, out_dir):
    """
    Download a file from a URL if it doesn't already exist locally.
    
    Parameters
    ----------
    url : str
        URL to download from
    filename : str
        Name of the file to save
    out_dir : str
        Output directory path
    """
    if not os.path.exists(out_dir + filename):
        print(url)
        print('downloading -- ', filename)

        response = requests.get(url)

        if response.status_code == 200:
            with open(out_dir + filename, 'wb') as file:
                file.write(response.content)


def download_cutout(sample, size=npix, out_dir='cutouts/'):
    """
    Download galaxy cutouts from the Legacy Survey.
    
    Parameters
    ----------
    sample : ndarray
        Array of shape (N, 2) containing RA and Dec coordinates
    size : int, optional
        Size of cutout in pixels (default: 128)
    out_dir : str, optional
        Output directory for cutout files (default: 'cutouts/')
    
    Notes
    -----
    Uses different Legacy Survey layers for north (Dec > 32.375) and south
    regions. Downloads are automatically skipped if files already exist.
    """
    for j in range(len(sample)):
        ra, dec = sample[j, 0], sample[j, 1]
        filename = 'cutout_{:.4f}_{:.4f}.fits'.format(ra, dec)

        # Use different survey layers for north and south
        if dec < 32.375:
            url = ('https://www.legacysurvey.org/viewer/fits-cutout?ra={:.4f}&dec={:.4f}&size=' +
                   str(size) + '&layer=ls-dr10-south&invvar&pixscale=0.262&bands=griz').format(ra, dec)
        else:
            url = ('https://www.legacysurvey.org/viewer/fits-cutout?ra={:.4f}&dec={:.4f}&size=' +
                   str(size) + '&layer=ls-dr9-north&invvar&pixscale=0.262&bands=grz').format(ra, dec)

        request_apply(url, filename, out_dir) 


def rgb_cutout(img_data):
    """
    Create an RGB image from multi-band cutout data.
    
    Parameters
    ----------
    img_data : ndarray
        Array of shape (3, npix, npix) with bands in order [i, r, g] or similar
    
    Returns
    -------
    rgb : ndarray
        RGB image array normalized with logarithmic stretch
    
    Notes
    -----
    Uses consistent intensity scaling across bands with z-band at 2x intensity.
    """
    imax = 0
    # Find maximum intensity across all bands
    for img in img_data[::-1]:
        val = np.percentile(img, 99.5)
        if val > imax:
            imax = val

    # Set up clipping intervals with manual normalization
    clip_intvls = 3 * [ManualInterval(vmin=0, vmax=imax)]
    clip_intvls[2] = ManualInterval(vmin=0, vmax=2*imax)

    return make_rgb(img_data[2], img_data[1], img_data[0],
                   interval=clip_intvls, stretch=LogStretch(a=20))

def image_bkg(img_dat):
    """
    Perform background subtraction and RMS estimation on multi-band image data.
    
    Parameters
    ----------
    img_dat : list
        List of FITS HDUs containing image data and inverse variance arrays
    
    Returns
    -------
    img_sub : ndarray
        Background-subtracted image data (4, npix, npix)
    img_hdr : FITS header
        Header from the image HDU
    img_rms : ndarray
        RMS noise maps for each band (4, npix, npix)
    glob_rms : ndarray
        Global median background level for each band (4,)
    flags : list
        Boolean flags indicating problematic bands
    
    Notes
    -----
    Uses 2D background estimation on source-masked images. Flags bands with
    no valid sources, inconsistent variance, or negative variance.
    """
    img_sub, img_rms = np.zeros((4, npix, npix)), np.zeros((4, npix, npix))
    glob_rms = np.zeros(4)
    flags = [False] * 4

    img_arr, img_hdr, img_invvar = img_dat[0].data, img_dat[0].header, img_dat[1].data

    w = wcs.WCS(img_hdr, naxis=2)

    # Handle case where z-band may be missing
    if np.shape(img_arr)[0] == 4:
        ixr, ir = range(4), range(4)
    else:
        flags[2] = True
        ixr, ir = range(3), [0, 1, 3]

    # Process each band
    for i, ix in zip(ir, ixr):
        # Detect and mask sources to isolate background
        threshold = detect_threshold(img_arr[ix, :, :], nsigma=1.0)
        segment_img = detect_sources(img_arr[ix, :, :], threshold, npixels=10, connectivity=8)

        try:
            mask = segment_img.make_source_mask(size=2)
        except:
            flags[i] = True
            continue

        # Convert inverse variance to variance
        var = np.reciprocal(img_invvar[ix, :, :])
        
        # Estimate 2D background
        bkg = Background2D(img_arr[ix, :, :].astype(img_arr[ix, :, :].dtype.newbyteorder('=')),
                          (32, 32), filter_size=(5, 5), mask=mask)

        # Check for problematic variance estimates
        if np.mean(var) > 100 or np.any(var < 0):
            flags[i] = True

        img_sub[i] = img_arr[ix, :, :] - np.abs(bkg.background)
        img_rms[i] = np.sqrt(var)
        glob_rms[i] = bkg.background_median

    return img_sub, img_hdr, img_rms, glob_rms, flags

def image_segmap(img_dat, img_rms, psf_mod, wc, flags):
    """
    Create segmentation maps identifying the target galaxy and companion sources.
    
    Parameters
    ----------
    img_dat : ndarray
        Background-subtracted image data (4, npix, npix)
    img_rms : ndarray
        RMS noise maps (4, npix, npix)
    psf_mod : ndarray
        PSF model for convolution
    wc : list
        [x, y] pixel coordinates of target center
    flags : list
        Boolean flags for problematic bands
    
    Returns
    -------
    comp_segmap : ndarray
        Binary segmentation map of target galaxy (npix, npix)
    comp_mask : ndarray
        Binary mask of companion/contaminating sources (npix, npix)
    flags : list
        Updated flags marking bands with segmentation failures
    
    Notes
    -----
    Identifies the brightest source near (wc[0], wc[1]) as the target galaxy.
    Companion mask includes all other detected sources.
    """
    segmaps_lo, segmaps_hi = np.zeros((4, npix, npix)), np.zeros((4, npix, npix))
    imzer = np.zeros((npix, npix))

    for iz, key in enumerate(bands):
        if flags[iz] == True:
            continue
        
        # Convolve with PSF model for source detection
        convolved_data = convolve(img_dat[iz], psf_mod)

        # Create segmentation map
        finder = SourceFinder(npixels=10, nlevels=32, contrast=0.0005, progress_bar=False)
        segment_map = finder(convolved_data, img_rms[iz])

        if segment_map == None:
            flags[iz] = True
            continue

        # Identify sources and find the target (brightest near center)
        cat = SourceCatalog(img_dat[iz], segment_map, convolved_data=convolved_data)
        tbl = cat.to_table()

        target = np.argmin(np.sqrt((tbl['xcentroid']-wc[0])**2 +
                                  (tbl['ycentroid']-wc[1])**2))
        
        # Create low and high flux segmentation maps
        segmaps_lo[iz] = (1 * (segment_map == (target+1)))
        segmaps_hi[iz] = (1 * ((segment_map != imzer) & (segment_map != (target+1))))

    # Combine across bands (g and r required)
    if not (flags[0] or flags[1]):
        comp_segmap = 1 * (np.logical_and(segmaps_lo[0], segmaps_lo[1]) &
                           (np.logical_xor(segmaps_lo[2], flags[2]) |
                            np.logical_xor(segmaps_lo[3], flags[3])))
        comp_mask = 1 * np.logical_or(np.logical_or(segmaps_hi[0], segmaps_hi[1]),
                                      np.logical_or(segmaps_hi[2], segmaps_hi[3]))
    else:
        flags = [True] * 4
    
    # Verify target is near center
    if comp_segmap[npix//2 - 1, npix//2 - 1] < 1:
        flags = [True] * 4

    return comp_segmap, comp_mask, flags

def plot_cutout(img_dat, comp_segmap, pos, w, ptr, df_row):
    """
    Create and save a visualization of the galaxy cutout with segmentation overlay.
    
    Parameters
    ----------
    img_dat : ndarray
        Background-subtracted image data (4, npix, npix)
    comp_segmap : ndarray
        Binary segmentation map of target galaxy
    pos : array
        [RA, Dec] coordinates in degrees
    w : WCS
        World coordinate system object
    ptr : ndarray
        Morphological parameters including centroid and size estimates
    df_row : pandas Series
        Source properties including logM, zspec, and color
    
    Notes
    -----
    Saves PNG to 'post_cutouts/galseg_{RA}_{DEC}.png' with overlaid ellipse
    and source information.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    img_copy = [img_dat[k].copy() for k in [0, 1, 3]]
    
    # Display full cutout
    ax.imshow(rgb_cutout(img_copy), origin='lower',
             extent=[pos[0]-16.506, pos[0]+16.506, pos[1]-16.506, pos[1]+16.506])

    # Overlay segmented galaxy
    for k in range(3):
        np.putmask(img_copy[k], ~(comp_segmap.astype('bool')), 0)
    ax.imshow(rgb_cutout(img_copy), origin='lower',
             extent=[pos[0]-16.506, pos[0]+16.506, pos[1]-16.506, pos[1]+16.506],
             alpha=0.5)

    # Draw ellipse overlay for Petrosian radius
    e = Ellipse(xy=w.pixel_to_world_values(ptr[0], ptr[1]),
               width=2*ptr[2]*pixsc, height=2*ptr[2]*(1-ptr[4])*pixsc,
               angle=ptr[3]*180/np.pi, lw=2, fill=False, color='azure', ls='--')
    ax.add_artist(e)

    # Add text annotations with source properties
    ax.text(pos[0]-16.5, pos[1]-14.5,
           r'${{\rm log}}M_{{\ast}}={0:.2f}$'.format(df_row['logM']),
           fontsize=30, color='white')
    ax.text(pos[0]+2.5, pos[1]-14.5,
           r'$z ={0:.2f}$'.format(df_row['zspec']),
           fontsize=30, color='white')
    ax.text(pos[0]+2.5, pos[1]+12.5,
           r'${{\it g-r}}={0:.2f}$'.format(df_row['g-r']),
           fontsize=30, color='white')

    ax.set_xlabel(r'$RA \ (deg)$', fontsize=40)
    ax.set_ylabel(r'$DEC \ (deg)$', fontsize=40)

    plt.savefig('post_cutouts/galseg_{:.4f}_{:.4f}.png'.format(pos[0], pos[1]),
               bbox_inches='tight')
    plt.close(fig)

def gaussian_2d(x, y, x0=0.0, y0=0.0, sigma=1.0):
    """
    2D Gaussian function.
    
    Parameters
    ----------
    x, y : float or ndarray
        Coordinates
    x0, y0 : float, optional
        Center coordinates (default: 0.0)
    sigma : float, optional
        Standard deviation (default: 1.0)
    
    Returns
    -------
    float or ndarray
        Gaussian value(s)
    """
    return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma**2))


def elliptical_transform(x, y, rp, th, e):
    """
    Transform Cartesian coordinates to elliptical radius.
    
    Parameters
    ----------
    x, y : float or ndarray
        Cartesian coordinates
    rp : float
        Semi-major axis (semi-major radius)
    th : float
        Position angle (rotation) in radians
    e : float
        Ellipticity (0-1, where 0 is circular)
    
    Returns
    -------
    float or ndarray
        Elliptical radius (1 = at the ellipse boundary)
    """
    return np.sqrt(np.square((x*np.cos(th)+y*np.sin(th))/rp) +
                   np.square((x*np.sin(th)-y*np.cos(th))/(rp*(1-e))))

def psf_grid(d_psf, fwhm, psf_type='gaussian'):
    """
    Generate a PSF model on a 2D grid.
    
    Parameters
    ----------
    d_psf : int
        Half-size of PSF grid (creates (2*d_psf+1) x (2*d_psf+1) grid)
    fwhm : float
        Full width at half maximum in pixels
    psf_type : str, optional
        Type of PSF model: 'gaussian' (default) or 'moffat'
    -
    Returns
    -------
    psf : ndarray
        Normalized PSF model
    """
    x = np.arange(-d_psf, d_psf+1)
    y = np.arange(-d_psf, d_psf+1)
    x_grid, y_grid = np.meshgrid(x, y)

    if psf_type == 'gaussian':
        psf = gaussian_2d(x_grid, y_grid, sigma=fwhm)
        return psf / np.sum(psf)
    elif psf_type == 'moffat':
        # Moffat profile with \rm{FWHM} = 2 \alpha \sqrt{2^{1 / \beta} - 1}
        alpha = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
        model = MoffatPSF(x_0=0.0, y_0=0.0, alpha=alpha, beta=2.0)
        psf = model(x_grid, y_grid)
        return psf / np.sum(psf)

def morph_main(sources, inds=None):
    """
    Main morphological analysis pipeline for a sample of galaxies.
    
    Parameters
    ----------
    sources : pandas DataFrame
        Source catalog with columns including 'ra', 'dec', 'ind', 'zspec',
        'logM', 'g-r', and PSF size estimates for each band (psfsize_g, etc.)
    inds : array-like, optional
        Indices to process. If None, process all sources.
    
    Returns
    -------
    morph_all : list
        List of morphological measurements. Each row contains:
        [source_ind, row_index, flattened_morph_arr]
        
        where morph_arr has shape (4, 19) for 4 bands with parameters:
        [0-7]: gini, m20, concentration, asymmetry, smoothness, shape_asymmetry,
               flag, snr_per_pixel
        [8]: r20/PSF size ratio
        [9]: segmentation to Petrosian ratio
        [10-14]: xc_asymmetry, yc_asymmetry, rpetro_ellip, orientation_asymmetry,
                 ellipticity_asymmetry
        [15-18]: sersic_n, sersic_rhalf, sersic_ellip, flag_sersic
    
    Notes
    -----
    For each galaxy:
    1. Downloads cutout from Legacy Survey
    2. Performs background subtraction
    3. Creates segmentation maps
    4. Computes morphological parameters using statmorph
    5. Saves visualization to post_cutouts/
    
    Failures at any stage are skipped (flagged).
    """
    morph_all = []
    model_psf = psf_grid(d_psf=8, fwhm=3)  # Generic PSF for source finding
    xx, yy = np.meshgrid(np.arange(0, npix, 1), np.arange(0, npix, 1))

    t1 = time.time()

    for ii, key in sources.iterrows():
        # Extract source coordinates
        pos = np.array([key['ra'], key['dec']])
        timestamp = np.round((time.time()-t1)/60.0, 3)
        print(ii, ',time;', timestamp, 'ind:', key['ind'], ',RA:', pos[0], ',DEC:', pos[1])

        # Download cutout from Legacy Survey
        download_cutout(pos.reshape(1, 2))
        filename = 'cutout_{:.4f}_{:.4f}.fits'.format(pos[0], pos[1])
        try:
            data = fits.open("cutouts/"+filename)
        except:
            continue

        # Background subtraction
        img_data, img_hdr, img_rms, glob_rms, flags = image_bkg(data)
        if (flags[0] or flags[1]):
            continue

        # Get WCS and convert to pixel coordinates
        w = wcs.WCS(img_hdr, naxis=2)
        pos_coord = SkyCoord(ra=pos[0]*u.degree, dec=pos[1]*u.degree, frame='icrs')
        wc = [int(i) for i in w.world_to_pixel(pos_coord)]

        # Source segmentation
        comp_segmap, comp_mask, flags = image_segmap(img_data, img_rms, model_psf, wc, flags)
        if (flags[0] or flags[1]):
            continue

        # Compute morphological parameters for each band
        morph_arr = np.zeros((4, 19))

        for jj, k in enumerate(bands):
            if flags[jj] == True:
                continue

            # Use Moffat PSF matching the actual observation
            obs_psf = psf_grid(d_psf=16, fwhm=key['psfsize_'+k]/pixsc, psf_type='moffat')
            
            try:
                # Compute morphological measurements using statmorph
                morph = statmorph.source_morphology(img_data[jj], comp_segmap,
                                                   weightmap=img_rms[jj],
                                                   mask=comp_mask.astype('bool'),
                                                   psf=obs_psf)[0]
            except:
                # Mark band as failed if morphology computation fails
                morph_arr[jj, 6], morph_arr[jj, -1] = 4, 4
                continue

            # Store basic morphological parameters
            morph_arr[jj, :8] = [morph.gini, morph.m20, morph.concentration,
                                morph.asymmetry, morph.smoothness, morph.shape_asymmetry,
                                morph.flag, morph.sn_per_pixel]

            # Store size-related metrics
            morph_arr[jj, 8] = morph.r20 / (key['psfsize_'+k]/pixsc)
            
            # Calculate segmentation fraction within Petrosian radius
            zz = elliptical_transform(xx-morph.xc_asymmetry, yy-morph.yc_asymmetry,
                                     morph.rpetro_ellip, morph.orientation_asymmetry,
                                     morph.ellipticity_asymmetry)
            morph_arr[jj, 9] = np.sum(1*(comp_segmap > 0)) / np.sum(1*(zz < 1))
            
            # Store asymmetry-related elliptical parameters
            morph_arr[jj, 10:15] = [morph.xc_asymmetry, morph.yc_asymmetry,
                                    morph.rpetro_ellip, morph.orientation_asymmetry,
                                    morph.ellipticity_asymmetry]

            # Store Sérsic profile parameters
            morph_arr[jj, 15:] = [morph.sersic_n, morph.sersic_rhalf,
                                 morph.sersic_ellip, morph.flag_sersic]

        # Create and save diagnostic plot
        plot_cutout(img_data, comp_segmap, pos, w, morph_arr[0, 10:15], key)

        # Append to results
        morph_row = [key['ind'], ii]
        morph_row.extend(morph_arr.ravel())
        morph_all.append(morph_row)

    t2 = time.time()
    print('Total time taken:', t2-t1)

    return morph_all
    

if __name__ == '__main__':
    """
    Main execution block: process galaxy sample and generate morphological measurements
    """
    # Load source catalog
    v1saga = pd.read_csv('v1saga_dwarfs.csv')
    print(v1saga.shape)

    # Run morphological analysis
    morph_all = morph_main(v1saga)

    # Define output column names
    param_names = ['gini', 'm20', 'c', 'a', 's', 'a_s', 'flag', 'snr_pix',
                   'r20_psf', 'segm_petr', 'xc_as', 'yc_as', 'rpetro',
                   'orien_as', 'ellip_as', 'n', 'rhalf', 'ellip_ss', 'flag_ss']
    col_names = ['ind0', 'ind1']
    for b in bands:
        col_names.extend([x+'_'+b for x in param_names])

    # Convert to DataFrame and save
    smorph = pd.DataFrame(morph_all, columns=col_names)
    smorph.to_csv('saga_morph.csv', index=False)
    print(smorph.shape)

    # Plot distribution of selected morphological parameters
    fig, ax = plt.subplots(1, 5, figsize=(30, 8), sharey=True)

    for i, key in enumerate(bands):
        # Select sources with high SNR and good flags
        high_snr = np.where((smorph['snr_pix_'+key] > 2) &
                           (smorph['flag_'+key] <= 1))[0]
        print(len(high_snr))

        # Plot first 5 morphological parameters
        for j in range(5):
            ax[j].hist(smorph[param_names[j]+'_'+key][high_snr],
                      bins=21, alpha=0.25, color=bands[key])
    
    plt.subplots_adjust(wspace=0.0)
    plt.show()
        
