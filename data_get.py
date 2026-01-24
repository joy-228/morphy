import numpy as np
import numpy.random as npr
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
#import pandas as pd
import astropy.io.fits as fits
import astropy.wcs as wcs
import requests
import os
import argparse
import csv
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.27)


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


def download_cutout(sample,  filters='griz',size=128,out_dir='cutouts/'):
    # will download the files from NERSC into the work_dir folder

    for j in range(len(sample)):
        ra,dec = sample[j,0],sample[j,1]
        filename = 'cutout_{:.4f}_{:.4f}.fits'.format(ra,dec)
        url = ('https://www.legacysurvey.org/viewer/fits-cutout?ra={:.4f}&dec={:.4f}&size='+str(size)+'&layer=ls-dr10-south&invvar&pixscale=0.262&bands='+filters).format(ra,dec)

        request_apply(url,filename,out_dir)        


def load_mask(brickname):
    out_dir='bricks/'

    url = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/' + brickname[:3] + '/' + brickname + '/'
    filename = 'legacysurvey-' + brickname + '-maskbits.fits.fz'

    request_apply(url+filename,filename,out_dir) 

    filename = out_dir + filename 
    dat = fits.open(filename)
    mask = dat[1].data
    mask_hdr = dat[1].header

    #mask = 1 is the edge of the brick which we can unmask
    mask[mask==1] = 0

    #treat all other mask values the same
    mask[mask>0] = 1
    
    return(mask,mask_hdr)

def load_tractor(brickname):
    out_dir='bricks/'

    url = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/tractor/' + brickname[:3] + '/'
    filename = 'tractor-' + brickname + '.fits'

    request_apply(url,filename,out_dir) 

    # will load tractor into a dataframe instead of fits table for ease
    filename = out_dir + filename
    dat = fits.open(filename)

    tract = pd.DataFrame({'objid':dat[1].data['objid'].byteswap().newbyteorder()})

    # will go through each of these and load into a column in the dataframe. have to 'swap the byte order' for some reason..
    keys = ['ra', 'dec', 'brickname', 'type', 'bx', 'by', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'ebv']

    for key in keys:
        tract[key] = dat[1].data[key].byteswap().newbyteorder()

    # need to correct for extinction. these coefficients are from the LS docs
    gext,rext,iext,zext = 3.214,2.165,1.592,1.211

    tract['A_g'] = tract['ebv'] * gext
    tract['A_r'] = tract['ebv'] * rext
    tract['A_i'] = tract['ebv'] * iext
    tract['A_z'] = tract['ebv'] * zext

    # shape parameters from the sersic fits
    tract['n'] = dat[1].data['sersic'].byteswap().newbyteorder()
    tract['re'] = dat[1].data['shape_r'].byteswap().newbyteorder()

    # need to convert the tractor 'e1'/'e2' ellipticity params into the usual ellip = 1-b/a definition.
    e1 = dat[1].data['shape_e1'].byteswap().newbyteorder()
    e2 = dat[1].data['shape_e2'].byteswap().newbyteorder()
    ee = np.sqrt(e1**2 + e2**2)
    ba = (1-ee) / (1+ee)
    tract['ell'] = 1-ba
    tract['pa'] = 0.5*np.arctan2(e2, e1) * 180 / np.pi
    tract.loc[tract['pa']<0, 'pa'] = tract.loc[tract['pa']<0, 'pa'] + 180

    # these "DUP" sources are junk for this purpose
    tract = tract[tract['type'] != 'DUP']

    # the '0' indicates it's corrected for reddening
    tract['m_g0'] = -2.5*np.log10(tract['flux_g']) + 22.5 - tract['A_g']
    tract['m_r0'] = -2.5*np.log10(tract['flux_r']) + 22.5 - tract['A_r']
    tract['m_i0'] = -2.5*np.log10(tract['flux_i']) + 22.5 - tract['A_i']
    tract['m_z0'] = -2.5*np.log10(tract['flux_z']) + 22.5 - tract['A_z']

    return(tract)

