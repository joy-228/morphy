import numpy as np
import numpy.random as npr
import astropy.coordinates as coord
import astropy.units as u
#import matplotlib.patches as patches
from astropy.coordinates import SkyCoord
#import pandas as pd
#import pdb
import astropy.io.fits as fits
import astropy.wcs as wcs
import requests
import os
#from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
#from fast3tree import fast3tree
#from scipy.interpolate import RectBivariateSpline as RBS
#from sersic import Sersic
#from astropy.visualization import make_lupton_rgb
import argparse
import csv
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.27)

class saga:
    def __init__(self,xprop): 
        self.ra = np.genfromtxt('saga-dr3-tableC2.txt',usecols=3)
        self.dec = np.genfromtxt('saga-dr3-tableC2.txt',usecols=4)
        self.rmag = np.genfromtxt('saga-dr3-tableC2.txt',usecols=5)
        self.sersic = np.genfromtxt('saga-dr3-tableC2.txt',usecols=12)
        self.g_r = np.genfromtxt('saga-dr3-tableC2.txt',usecols=7)
        self.zspec = np.genfromtxt('saga-dr3-tableC2.txt',usecols=14)
        self.N = len(self.rmag)

        if 'reffr' in xprop: self.reffr = np.genfromtxt('saga-dr3-tableC2.txt',usecols=8)
        if 'sfbr' in xprop: self.sfbr = np.genfromtxt('saga-dr3-tableC2.txt',usecols=9)
        if 'mst' in xprop: self.mst = np.genfromtxt('saga-dr3-tableC2.txt',usecols=15)
        if 'dst' in xprop:
                zdist = cosmo.comoving_distance(zspec)
                self.dst = np.log10(zdist.value)

    def SF_selc(self,g_r_cut=0.6,zspec_cut=0.1):
        avail = np.where((self.g_r<g_r_cut)&(self.zspec<zspec_cut))[0]
        print(len(avail))
        file = open('saga_dwarfs.csv', 'w+', newline ='')
         
        # writing the data into the file
        with file:    
            write = csv.writer(file)
            write.writerow(["RA","DEC"])
            write.writerows([[self.ra[j],self.dec[j]] for j in avail])

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


def get_psf_model(ra, dec, tractor, LS_mask, images, filters, hdr):
    print('\nfinding PSF model.... ')
    # will get the psf model from some isolated star near the coords (ra, dec)

    # find some stars that are unsaturated and bright
    stars = tractor[(tractor['type']=='PSF') & (tractor['m_g0']<20) & (tractor['m_g0']>17.5) & (tractor['flux_r']>0) & (tractor['flux_z']>0)].copy()


    # find the ones that are isolated
    # we use fast3tree to do this proximity check a lot faster than a double-for-loop
    stars['isolated'] = False
    stars['masked'] = True
    coords = tractor[['bx', 'by']].to_numpy()
    with fast3tree(coords) as tree:
        for ii in range(len(stars)):
            cds = [stars['bx'].values[ii], stars['by'].values[ii]]
            rad = 20 #pixels, this is the isolation criterion
            idx = tree.query_radius(cds, rad)
            stars.iloc[ii, stars.columns.get_loc('isolated')] = len(idx) == 1


    # also should be some distance from a masked pix
    masked_pix = np.where(LS_mask > 1)
    masked_pix = np.vstack((masked_pix[1], masked_pix[0])).T
    with fast3tree(masked_pix) as tree:
        for ii in range(len(stars)):
            cds = [stars['bx'].values[ii], stars['by'].values[ii]]
            rad = 10 #pixels; this is the isolation criterion from a masked pix
            idx = tree.query_radius(cds, rad)
            stars.iloc[ii, stars.columns.get_loc('masked')] = len(idx) > 0


    stars = stars[(stars['isolated'] == True) & (stars['masked'] == False)]
    print(len(stars), ' isolated stars found')


    # we need to find the x, y location of the ra, dec coords
    w = wcs.WCS(hdr)
    pix_coords = w.all_world2pix([[ra, dec]], 1)[0]

    # now we find the closest isolated star to this location and will use that as the PSF model
    coords = stars[['bx', 'by']].to_numpy()
    dist = np.sqrt((coords[:,0]-pix_coords[0])**2 + (coords[:,1]-pix_coords[1])**2)
    nearest = np.argmin(dist)
    print('nearest at, ', coords[nearest])
    d_psf = 12 # in pixels, the size of the cutout


    # and we loop through the filters to get a cutout/psf model for each filter band
    psfs = []
    for ii in range(len(filters)):

        model = images[ii][(round(coords[nearest, 1])-d_psf):(round(coords[nearest, 1])+d_psf+1), (round(coords[nearest, 0])-d_psf):(round(coords[nearest, 0])+d_psf+1)]
        model = np.copy(model)

        # now we center the psf model via interpolation
        x = np.arange(-d_psf, d_psf+1)
        y = np.arange(-d_psf, d_psf+1)
        model_interp = RBS(y, x, model)

        xcen = coords[nearest, 0] - round(coords[nearest, 0])
        ycen = coords[nearest, 1] - round(coords[nearest, 1])

        xi = np.arange(xcen-d_psf+1, xcen+d_psf)
        yi = np.arange(ycen-d_psf+1, ycen+d_psf)

        model_2 = model_interp(yi, xi)
        model_2 /= np.sum(model_2)

        psfs.append(model_2)

    return(psfs)

def model_and_subtract_tractor_sources(tractor, images, psfs, LS_mask, filters, work_dir, brickname):
    # will subtract (and mask) the tractor sources that we feel we can subtract safely. this is three different sources:
    # all PSF sources
    # all red sources
    # all high-ish surface brightness sources that are probably not shredded

    # anything redder than this will be modelled, subtracted out, and masked. might need to tweak if the LSBG is expected to be redder.
    g_r_color_threshold = 0.5


    # anything higher in surface brightness that this (defined as avg SB within the effective radius) or brighter than this in m_g0 will be assumed impervious to shredding and that the tractor measurements is correct.
    sb_cut = 22
    mg_cut = 18


    # models will hold the images of the tractor models once we create them. now we just set them up
    models = []
    for ii, ff in enumerate(filters):
        models.append(np.zeros(np.shape(images[ii])))



    ###########################################################################################
    ### first we subtract out all stars
    stars = tractor[(tractor['type']=='PSF')]
    print('loading stars.. ')
    for ii in range(len(stars)):
        if ii % 100 == 0:
            print(ii, ' / ', len(stars))
        # need to allocate flux to the four pixels the center is inbetween since the star won't line up with the pixel griod
        x1, y1 = int(np.floor(stars['bx'].values[ii])), int(np.floor(stars['by'].values[ii]))
        x2 = x1 + 1
        y2 = y1 + 1
        xcen = stars['bx'].values[ii]
        ycen = stars['by'].values[ii]

        # will do this with interp2d for ease
        tmp_image = np.zeros((3,3))
        tmp_image[1,1] = 1
        x, y = np.arange(xcen-x1-1, xcen-x1+2), np.arange(ycen-y1-1, ycen-y1+2)
        model = RBS(y, x, tmp_image, kx=1, ky=1)

        xi = [0, 1]
        yi = [0, 1]
        resampled = model(yi, xi)

        # need to trim in case near the edge of the image
        x1t = max([0, x1])
        y1t = max([0, y1])
        x2t = min([np.shape(models[0])[1]-1, x2])
        y2t = min([np.shape(models[0])[0]-1, y2])

        if x1t > x1:
            resampled = resampled[:, (x1t-x1):]
        if y1t > y1:
            resampled = resampled[(y1t-y1):, :]
        if x2t < x2:
            resampled = resampled[:, :-(x2-x2t)]
        if y2t < y2:
            resampled = resampled[:-(y2-y2t), :]


        for jj, ff in enumerate(filters):
            key = 'flux_' + ff
            models[jj][y1t:(y2t+1), x1t:(x2t+1)] += stars[key].values[ii] * resampled

    print('... loaded stars')

    ###########################################################################################
    ###
    # we do the red-things and high-ish surface brightness things together
    # we do a more conservative cut here on g-r and do a more stringent cut later on with the sersic photometry.

    red_things = tractor[(tractor['type']!='PSF') & (tractor['m_g0']-tractor['m_r0'] > g_r_color_threshold)]

    ### hi SB things

    # need to add this column into tractor. this is avg SB within the effective radius
    tractor['mu_eff_g'] = tractor['m_g0'] + 2.5*np.log10(2*np.pi*tractor['re']**2*(1-tractor['ell']))

    hsb = tractor[(tractor['type']!='PSF') & ~np.isin(tractor.objid, red_things.objid) & ((tractor['mu_eff_g'] < sb_cut) | (tractor['m_g0'] < mg_cut))]


    # combine them together to loop through all at once
    red_things = pd.concat([red_things, hsb], sort=False)
    red_things = red_things.sort_values('m_r0', ascending=True)



    print('loading red and hsb things.. ')
    ###### model these together:
    for ii in range(len(red_things)):
        if ii % 100 == 0:
            print(ii, ' / ', len(red_things))

        dim = int(6*red_things['re'].values[ii])

        # high sersic-index sources have big outskirts so need to model further out
        if red_things['n'].values[ii] > 3:
            dim *= 3

        #also bright things need to be modelled further out
        if red_things['m_g0'].values[ii] < 18:
            dim *= 3


        offx = red_things['bx'].values[ii] - int(red_things['bx'].values[ii])
        offy = red_things['by'].values[ii] - int(red_things['by'].values[ii])
        ser_params = {'X0': dim+offx, 'Y0': dim+offy,  'PA': red_things['pa'].values[ii], 'ell': red_things['ell'].values[ii],  'n': red_things['n'].values[ii],  'I_e': 0.1, 'r_e': red_things['re'].values[ii]/px_sc}
        sers = Sersic(ser_params)

        img = sers.array((2*dim,2*dim))
        img /= np.sum(img)


        x1 = int(red_things['bx'].values[ii]) - dim + 1
        x2 = int(red_things['bx'].values[ii]) + dim + 1
        y1 = int(red_things['by'].values[ii]) - dim + 1
        y2 = int(red_things['by'].values[ii]) + dim + 1

        # need to trim in case near the edge of the image
        x1t = max([0, x1])
        y1t = max([0, y1])
        x2t = min([np.shape(models[0])[1], x2])
        y2t = min([np.shape(models[0])[0], y2])

        if x1t > x1:
            img = img[:, (x1t-x1):]
        if y1t > y1:
            img = img[(y1t-y1):, :]
        if x2t < x2:
            img = img[:, :-(x2-x2t)]
        if y2t < y2:
            img = img[:-(y2-y2t), :]


        # add this object into the model images
        for jj, ff in enumerate(filters):
            key = 'flux_' + ff
            models[jj][y1t:y2t, x1t:x2t] += red_things[key].values[ii] * img


        #if red_things['objid'].values[ii] == 1317: pdb.set_trace()




    ##############################################################################
    ## Now we move to creating a mask. again we mask three things:
    # stars
    # HSB-ish things
    # all red things


    ##first we mask the HSB things and all the red things. We can do these together
    hsb = tractor[((tractor['mu_eff_g'] < sb_cut) | (tractor['m_g0'] < mg_cut)) | (tractor['m_g0']-tractor['m_r0'] > g_r_color_threshold)]

    print('masking all hsb things ....')

    tractor_mask = np.zeros(np.shape(models[0]))
    xx, yy = np.meshgrid(range(np.shape(tractor_mask)[1]), range(np.shape(tractor_mask)[0]))
    xx, yy = xx.flatten(), yy.flatten()
    coords = np.vstack((xx, yy)).T
    with fast3tree(coords) as tree:
        for ii in range(len(hsb)):
            if ii % 100 == 0:
                print(ii, ' / ', len(hsb))

            cds = [hsb['bx'].values[ii], hsb['by'].values[ii]]
            re = hsb['re'].values[ii]
            #we mask all pixels within some radius, rad, of this 'hsb' object. the radius is ~ the r_e of the tractor source, with some tweaking from trial-and-error
            if re > 0 and re < 2:
                re = 2
            rad = 2 * re / px_sc
            if hsb['mu_eff_g'].values[ii] > 24 or hsb['m_g0'].values[ii] > 22: # too likely these are shreds so won't mask
                rad = 0
            elif hsb['mu_eff_g'].values[ii] > sb_cut and hsb['m_g0'].values[ii] > mg_cut:
                rad = re /  px_sc

            # if the tractor source is some high-sersic-index source (i.e. with big envelope), we mask further out
            elif hsb['n'].values[ii] > 4:
                rad = 2 * re / px_sc

            # in this case, this is HSB source is just a star
            if rad == 0:
                # this is from the LS webpage
                rad = 1630. * 1.396**(-hsb['m_g0'].values[ii]) / px_sc

            idx = tree.query_radius(cds, rad)
            tractor_mask[yy[idx], xx[idx]] = 1



    ##############################################################################
    ## will make a separate star mask for everything that tractor thinks is a star.

    print('masking all star things ....')

    star_mask = np.zeros(np.shape(models[0]))
    xx, yy = np.meshgrid(range(np.shape(tractor_mask)[1]), range(np.shape(tractor_mask)[0]))
    xx, yy = xx.flatten(), yy.flatten()
    coords = np.vstack((xx, yy)).T
    with fast3tree(coords) as tree:
        for ii in range(len(stars)):
            if ii % 100 == 0:
                print(ii, ' / ', len(stars))

            cds = [stars['bx'].values[ii], stars['by'].values[ii]]
            rad = 1630. * 1.396**(-stars['m_g0'].values[ii]) / px_sc
            if np.isnan(rad):
                continue
            idx = tree.query_radius(cds, rad)
            star_mask[yy[idx], xx[idx]] = 1



    # to finish the models, we need to convolve with the PSFs
    for ii, ff in enumerate(filters):
        models[ii] = convolve_with_psf(models[ii], psfs[ii])


    # we create the resids images by subtracting out the tractor model
    resids = []
    for ii, ff in enumerate(filters):
        resids.append(images[ii] - models[ii])


    # now we combine the masks into one: raw LS_mask, tractor_mask, and star_mask
    LS_mask[star_mask > 0] = 1
    LS_mask[tractor_mask > 0] = 1


    '''
    print('saving the model and resid images.. ')
    for ii, ff in enumerate(filters):
        filename = work_dir + brickname + '_'+ff+'_model.fits'
        dd = fits.PrimaryHDU(data=models[ii])
        dd.writeto(filename, overwrite=True)

        filename = work_dir + brickname + '_'+ff+'_resid.fits'
        dd = fits.PrimaryHDU(data=resids[ii])
        dd.writeto(filename, overwrite=True)
    '''

    return(LS_mask)
                


if __name__ == '__main__':
    #ra, dec = 304.8319, -63.6431
    #brickname = '3045m637'
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--objid',required=True,
                        help='coadd to grab')
    args = parser.parse_args()
    y6_lsbgs = fits.open('/Users/f006zcq/Desktop/DES Y6 LSBGs/GalfitTweaks/y6_gold_2_0_lsbg_v2.fits')[1].data
    obj = y6_lsbgs[y6_lsbgs['COADD_OBJECT_ID'] == int(args.objid)]

    bricks = fits.open('/Users/f006zcq/Downloads/survey-bricks.fits')[1].data
    sel = (bricks['DEC1'] < obj['DEC']) & (bricks['DEC2'] > obj['DEC']) & (bricks['RA1'] < obj['RA']) & (bricks['RA2'] > obj['RA'])
    
    

    ra, dec = obj['RA'][0], obj['DEC'][0]
    brickname = bricks[sel]['BRICKNAME'][0]


    filters = ['g', 'r', 'i']

    work_dir = '/Users/f006zcq/Desktop/DES Y6 LSBGs/Tractor_LSBG/test/'


    # download the legacy survey data (tractor catalog, g/r/z data, builtin masks)
    download_data(brickname, work_dir, filters)

    # load the tractor catalog
    tractor = load_tractor(brickname, work_dir)


    # load the brick images and a header for WCS info
    images, hdr = load_images(brickname, work_dir, filters)

    # load the  pre-made legacysurvey mask (which at least masks the brightest stars, but not much else)
    LS_mask = load_mask(brickname, work_dir)


    # to create the tractor model, we need a PSF model which we just take as a nearby isolated star
    psfs = get_psf_model(ra, dec, tractor, LS_mask, images, filters, hdr)


    # now we are ready to find, model, subtract, and mask the appropriate tractor sources
    resids, models, mask = model_and_subtract_tractor_sources(tractor, images, psfs, LS_mask, filters, work_dir, brickname)

    # save mask for kai lol
    fits.writeto(f'{args.objid}_mask_{brickname}.fits',mask)



    # plot things
    plot_images_and_mask(images, resids, models, mask, hdr, ra, dec)


def model_and_subtract_tractor_sources(tractor, images, psfs, LS_mask, filters, work_dir, brickname):
    # will subtract (and mask) the tractor sources that we feel we can subtract safely. this is three different sources:
    # all PSF sources
    # all red sources
    # all high-ish surface brightness sources that are probably not shredded

    # anything redder than this will be modelled, subtracted out, and masked. might need to tweak if the LSBG is expected to be redder.
    g_r_color_threshold = 1.1


    # anything higher in surface brightness that this (defined as avg SB within the effective radius) or brighter than this in m_g0 will be assumed impervious to shredding and that the tractor measurements is correct.
    sb_cut = 22
    mg_cut = 19


    # models will hold the images of the tractor models once we create them. now we just set them up
    models = []
    for ii, ff in enumerate(filters):
        models.append(np.zeros(np.shape(images[ii])))



    ###########################################################################################
    ### first we subtract out all stars
    stars = tractor[(tractor['type']=='PSF')]
    print('loading stars.. ')
    for ii in range(len(stars)):
        if ii % 100 == 0:
            print(ii, ' / ', len(stars))
        # need to allocate flux to the four pixels the center is inbetween since the star won't line up with the pixel griod
        x1, y1 = int(np.floor(stars['bx'].values[ii])), int(np.floor(stars['by'].values[ii]))
        x2 = x1 + 1
        y2 = y1 + 1
        xcen = stars['bx'].values[ii]
        ycen = stars['by'].values[ii]

        # will do this with interp2d for ease
        tmp_image = np.zeros((3,3))
        tmp_image[1,1] = 1
        x, y = np.arange(xcen-x1-1, xcen-x1+2), np.arange(ycen-y1-1, ycen-y1+2)
        model = RBS(y, x, tmp_image, kx=1, ky=1)

        xi = [0, 1]
        yi = [0, 1]
        resampled = model(yi, xi)

        # need to trim in case near the edge of the image
        x1t = max([0, x1])
        y1t = max([0, y1])
        x2t = min([np.shape(models[0])[1]-1, x2])
        y2t = min([np.shape(models[0])[0]-1, y2])

        if x1t > x1:
            resampled = resampled[:, (x1t-x1):]
        if y1t > y1:
            resampled = resampled[(y1t-y1):, :]
        if x2t < x2:
            resampled = resampled[:, :-(x2-x2t)]
        if y2t < y2:
            resampled = resampled[:-(y2-y2t), :]


        for jj, ff in enumerate(filters):
            key = 'flux_' + ff
            models[jj][y1t:(y2t+1), x1t:(x2t+1)] += stars[key].values[ii] * resampled




    print('... loaded stars')



    ###########################################################################################
    ###
    # we do the red-things and high-ish surface brightness things together
    # we do a more conservative cut here on g-r and do a more stringent cut later on with the sersic photometry.

    red_things = tractor[(tractor['type']!='PSF') & (tractor['m_g0']-tractor['m_r0'] > g_r_color_threshold)]

    ### hi SB things

    # need to add this column into tractor. this is avg SB within the effective radius
    tractor['mu_eff_g'] = tractor['m_g0'] + 2.5*np.log10(2*np.pi*tractor['re']**2*(1-tractor['ell']))

    hsb = tractor[(tractor['type']!='PSF') & ~np.isin(tractor.objid, red_things.objid) & ((tractor['mu_eff_g'] < sb_cut) | (tractor['m_g0'] < mg_cut))]


    # combine them together to loop through all at once
    red_things = pd.concat([red_things, hsb], sort=False)
    red_things = red_things.sort_values('m_r0', ascending=True)



    print('loading red and hsb things.. ')
    ###### model these together:
    for ii in range(len(red_things)):
        if ii % 100 == 0:
            print(ii, ' / ', len(red_things))

        dim = int(6*red_things['re'].values[ii])

        # high sersic-index sources have big outskirts so need to model further out
        if red_things['n'].values[ii] > 3:
            dim *= 3

        #also bright things need to be modelled further out
        if red_things['m_g0'].values[ii] < 18:
            dim *= 3


        offx = red_things['bx'].values[ii] - int(red_things['bx'].values[ii])
        offy = red_things['by'].values[ii] - int(red_things['by'].values[ii])
        ser_params = {'X0': dim+offx, 'Y0': dim+offy,  'PA': red_things['pa'].values[ii], 'ell': red_things['ell'].values[ii],  'n': red_things['n'].values[ii],  'I_e': 0.1, 'r_e': red_things['re'].values[ii]/px_sc}
        sers = Sersic(ser_params)

        img = sers.array((2*dim,2*dim))
        img /= np.sum(img)


        x1 = int(red_things['bx'].values[ii]) - dim + 1
        x2 = int(red_things['bx'].values[ii]) + dim + 1
        y1 = int(red_things['by'].values[ii]) - dim + 1
        y2 = int(red_things['by'].values[ii]) + dim + 1

        # need to trim in case near the edge of the image
        x1t = max([0, x1])
        y1t = max([0, y1])
        x2t = min([np.shape(models[0])[1], x2])
        y2t = min([np.shape(models[0])[0], y2])

        if x1t > x1:
            img = img[:, (x1t-x1):]
        if y1t > y1:
            img = img[(y1t-y1):, :]
        if x2t < x2:
            img = img[:, :-(x2-x2t)]
        if y2t < y2:
            img = img[:-(y2-y2t), :]


        # add this object into the model images
        for jj, ff in enumerate(filters):
            key = 'flux_' + ff
            models[jj][y1t:y2t, x1t:x2t] += red_things[key].values[ii] * img


        #if red_things['objid'].values[ii] == 1317: pdb.set_trace()




    ##############################################################################
    ## Now we move to creating a mask. again we mask three things:
    # stars
    # HSB-ish things
    # all red things


    ##first we mask the HSB things and all the red things. We can do these together
    hsb = tractor[((tractor['mu_eff_g'] < sb_cut) | (tractor['m_g0'] < mg_cut)) | (tractor['m_g0']-tractor['m_r0'] > g_r_color_threshold)]

    print('masking all hsb things ....')

    tractor_mask = np.zeros(np.shape(models[0]))
    xx, yy = np.meshgrid(range(np.shape(tractor_mask)[1]), range(np.shape(tractor_mask)[0]))
    xx, yy = xx.flatten(), yy.flatten()
    coords = np.vstack((xx, yy)).T
    with fast3tree(coords) as tree:
        for ii in range(len(hsb)):
            if ii % 100 == 0:
                print(ii, ' / ', len(hsb))

            cds = [hsb['bx'].values[ii], hsb['by'].values[ii]]
            re = hsb['re'].values[ii]
            #we mask all pixels within some radius, rad, of this 'hsb' object. the radius is ~ the r_e of the tractor source, with some tweaking from trial-and-error
            if re > 0 and re < 2:
                re = 2
            rad = 2 * re / px_sc
            if hsb['mu_eff_g'].values[ii] > 24 or hsb['m_g0'].values[ii] > 22: # too likely these are shreds so won't mask
                rad = 0
            elif hsb['mu_eff_g'].values[ii] > sb_cut and hsb['m_g0'].values[ii] > mg_cut:
                rad = re /  px_sc

            # if the tractor source is some high-sersic-index source (i.e. with big envelope), we mask further out
            elif hsb['n'].values[ii] > 4:
                rad = 2 * re / px_sc

            # in this case, this is HSB source is just a star
            if rad == 0:
                # this is from the LS webpage
                rad = 1630. * 1.396**(-hsb['m_g0'].values[ii]) / px_sc

            idx = tree.query_radius(cds, rad)
            tractor_mask[yy[idx], xx[idx]] = 1



    ##############################################################################
    ## will make a separate star mask for everything that tractor thinks is a star.

    print('masking all star things ....')

    star_mask = np.zeros(np.shape(models[0]))
    xx, yy = np.meshgrid(range(np.shape(tractor_mask)[1]), range(np.shape(tractor_mask)[0]))
    xx, yy = xx.flatten(), yy.flatten()
    coords = np.vstack((xx, yy)).T
    with fast3tree(coords) as tree:
        for ii in range(len(stars)):
            if ii % 100 == 0:
                print(ii, ' / ', len(stars))

            cds = [stars['bx'].values[ii], stars['by'].values[ii]]
            rad = 1630. * 1.396**(-stars['m_g0'].values[ii]) / px_sc
            if np.isnan(rad):
                continue
            idx = tree.query_radius(cds, rad)
            star_mask[yy[idx], xx[idx]] = 1



    # to finish the models, we need to convolve with the PSFs
    for ii, ff in enumerate(filters):
        models[ii] = convolve_with_psf(models[ii], psfs[ii])


    # we create the resids images by subtracting out the tractor model
    resids = []
    for ii, ff in enumerate(filters):
        resids.append(images[ii] - models[ii])


    # now we combine the masks into one: raw LS_mask, tractor_mask, and star_mask
    LS_mask[star_mask > 0] = 1
    LS_mask[tractor_mask > 0] = 1



    print('saving the model and resid images.. ')
    for ii, ff in enumerate(filters):
        filename = work_dir + brickname + '_'+ff+'_model.fits'
        dd = fits.PrimaryHDU(data=models[ii])
        dd.writeto(filename, overwrite=True)

        filename = work_dir + brickname + '_'+ff+'_resid.fits'
        dd = fits.PrimaryHDU(data=resids[ii])
        dd.writeto(filename, overwrite=True)



    return(resids, models, LS_mask)





