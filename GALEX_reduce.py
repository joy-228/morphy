"""
GALEX NUV Aperture Photometry Pipeline
=======================================
Uses photutils to measure GALEX NUV fluxes for galaxies from a user-provided catalog.

Pipeline steps:
  1. Download 5'x5' NUV cutouts from the DESI Legacy Sky Viewer
  2. Mask all catalog sources (r < 21) within 2 effective radii (or 2x GALEX PSF if small)
  3. Estimate background (median) and variance from unmasked pixels
  4. Perform elliptical (or circular 6") aperture photometry
  5. Flag non-detections (S/N < 4) and missing data

Input catalog columns expected (FITS or CSV):
    - ra, dec              : sky coordinates (degrees)
    - r_eff                : effective radius (arcsec)
    - ellipticity          : 1 - b/a  (0 = circular)
    - position_angle       : degrees East of North
    - rmag                : r-band magnitude (for masking)

Usage:
    python galex_nuv_photometry.py --catalog my_catalog.fits --output results.fits
    python galex_nuv_photometry.py --catalog my_catalog.csv  --output results.csv
"""

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
#from astropy.wcs.utils import skycoord_to_pixel

from photutils.aperture import EllipticalAperture,CircularAperture,aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog

from dl import storeClient as sc
from data_get import *

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CUTOUT_SIZE_ARCMIN = 5.0        # requested image size [arcmin]
CUTOUT_HALF = 0.5*CUTOUT_SIZE_ARCMIN
SN_THRESHOLD = 4.0              # S/N threshold for non-detection flag
MASK_R_MAG_LIMIT = 21.0        # mask sources brighter than this
PIX_SC = 1.5        # GALEX NUV pixel scale [arcsec/pixel]
DEC_PIX_SC = 0.262 # DECam pixel scale [arcsec/pixel]
PSF_FWHM = 6.0 / PIX_SC         # GALEX NUV FWHM [pix]
# NUV gain for S/N estimation (counts s⁻¹ → electrons); adjust if using raw data
GALEX_NUV_GAIN = 1.0
GALEX_NUV_ZP_AB = 20.08   # Morrissey et al. 2007
# Salpeter -> Koupa IMF using Madua & Dickinson 2014 (Figure 4)
_IMF_FACTOR = 0.66

# ── DESI Legacy Sky Viewer downloader ───────────────────────────────────────

def download_galex_cutout(
    pos: [float,float],
    out_dir: str = "GALEX_cutouts/",
) -> tuple[np.ndarray | None, WCS | None]:
    """
    Download a NUV FITS cutout from the DESI Legacy Sky Viewer.

    Parameters
    ----------
    pos      : Target position in ra, dec [deg,deg]
    out_dir      : Directory to save cutouts (default: "GALEX_cutouts/")    
    Returns
    -------
    data : 2-D numpy array (counts/s) or None if download fails
    wcs  : astropy WCS or None
    """
    size_pix = int(round(CUTOUT_SIZE_ARCMIN * 60.0 / PIX_SC))
    filename = 'cutout_{:.6f}_{:.6f}.fits'.format(pos[0], pos[1])
    url = (
        "https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={pos[0]:.6f}&dec={pos[1]:.6f}"
        f"&size={size_pix}&layer=galex&pixscale={PIX_SC}&bands=n"
    )
    try:
        resp = requests.get(url,filename,out_dir, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Download failed for (%.4f, %.4f): %s", pos[0], pos[1], exc)
        return None, None

    with fits.open(BytesIO(resp.content)) as hdul:
        # Legacy viewer returns primary HDU with image + WCS
        if hdul[0].data is None or hdul[0].data.ndim < 2:
            return None, None
        data = hdul[0].data.astype(np.float64)
        if data.ndim == 3:          # multi-band cube → take first (NUV)
            data = data[0]
        wcs = WCS(hdul[0].header, naxis=2)
    return data, wcs


# ── Background estimation ────────────────────────────────────────────────────

def build_source_mask(
    data: np.ndarray,
    w: WCS,
    tractor_catalog: pd.DataFrame,
) -> np.ndarray:
    """
    Build a boolean mask (True = masked) for all catalog sources with r < MASK_R_MAG_LIMIT.
    Each source is masked inside an ellipse of radius = max(2*r_eff, 2*PSF).

    Parameters
    ----------
    data             : Image array (ny, nx)
    wcs              : Image WCS
    tractor_catalog  : DataFrame with columns ra, dec, r_eff, ellipticity,
                       position_angle, r_mag

    Returns
    -------
    mask : bool array, shape (ny, nx), True where pixels are masked
    """
    ny, nx = data.shape
    yy, xx = np.ogrid[:ny, :nx]
    mask = np.zeros((ny, nx), dtype=bool)


    for _, src in data.iterrows():
        # Convert sky → pixel
        sky = SkyCoord(ra=src["ra"] * u.deg, dec=src["dec"] * u.deg)
        try:
            xc, yc = w.world_to_pixel(sky)
        except Exception:
            continue

        r_eff_pix = src['shape_r'] / PIX_SC          # effective radius [pix]
        r_mask    = max(2.0 * r_eff_pix, 2.0 * PSF_FWHM) # masking radius [pix]

        # Ellipse axis ratio
        b_over_a,pa_rad = src["ba"],np.deg2rad(src["PA"])

        # Rotated ellipse equation
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        dx,dy = xx - xc, yy - yc
        x_rot,y_rot =  dx * cos_pa + dy * sin_pa, -dx * sin_pa + dy * cos_pa

        inside = (x_rot / r_mask) ** 2 + (y_rot / (r_mask * b_over_a)) ** 2 <= 1.0
        mask |= inside

    fig,ax=plt.subplots(1,2,figsize=(16,8),sharey=True,sharex=True)

    ax[0].imshow(data, interpolation='nearest', origin='lower')
    ax[1].imshow(mask, origin='lower')
    plt.show()

    return mask


def estimate_background(
    data: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """
    Estimate background level and pixel variance from unmasked pixels.

    Returns
    -------
    bkg_median : median of unmasked pixels
    bkg_var    : variance of unmasked pixels
    """
    unmasked = data[~mask]
    if unmasked.size == 0:
        log.warning("No unmasked background pixels — returning zeros.")
        return 0.0, 0.0
    bkg_median = float(np.median(unmasked))
    bkg_var    = float(np.var(unmasked, ddof=1))
    return bkg_median, bkg_var


# ── Aperture photometry ──────────────────────────────────────────────────────

def do_aperture_photometry(
    data: np.ndarray,
    w: WCS,
    pos: tuple[float, float],
    r_eff: float,
    b_over_a: float,
    position_angle: float,
    bkg_median: float,
    bkg_var: float,
) -> dict:
    """
    Perform aperture photometry on one target.

    Uses an elliptical aperture from optical photometry, or a circular 6"
    aperture when r_eff < GALEX PSF.

    Returns
    -------
    dict with keys:
        flux_counts   : background-subtracted flux [image units × pix²]
        flux_err      : photometric uncertainty
        snr           : signal-to-noise ratio
        aperture_type : 'elliptical' or 'circular'
        a_arcsec      : semi-major axis of aperture used [arcsec]
        flag_nondet   : True if S/N < 4
        flag_nodata   : True if no GALEX data at position
    """
    result = dict(
        flux_counts=np.nan,flux_err=np.nan,
        snr=np.nan,aperture_type="none",a_arcsec=np.nan,
        flag_nondet=False,flag_nodata=False,
    )

    # ── Check data availability ──────────────────────────────────────────────
    sky = SkyCoord(ra=pos[0] * u.deg, dec=pos[1] * u.deg)
    try:
        xc, yc = w.world_to_pixel(sky)
    except Exception:
        result["flag_nodata"] = True
        return result

    ny, nx = data.shape
    if not (0 <= xc < nx and 0 <= yc < ny):
        result["flag_nodata"] = True
        return result

    # ── Build aperture ───────────────────────────────────────────────────────
    r_eff_pix = r_eff / PIX_SC

    if r_eff_pix < PSF_FWHM:
        # Use circular 6" aperture
        aperture = CircularAperture((xc, yc), r=PSF_FWHM)
        result["aperture_type"] = "circular"
        result["a_arcsec"]      = PSF_FWHM
    else:
        # Use elliptical aperture from optical photometry
        # photutils EllipticalAperture: theta is CCW from +x axis
        theta    = np.deg2rad(90.0 - position_angle)  # PA (E of N) → theta
        aperture = EllipticalAperture(
            (xc, yc), a=r_eff_pix, b=r_eff_pix * b_over_a, theta=theta
        )
        result["aperture_type"] = "elliptical"
        result["a_arcsec"]      = r_eff

    # ── Aperture photometry ──────────────────────────────────────────────────
    bkg_sub_data = data - bkg_median
    phot_table   = aperture_photometry(bkg_sub_data, aperture)
    raw_flux     = float(phot_table["aperture_sum"][0])

    # ── Uncertainty estimate ─────────────────────────────────────────────────
    # σ² = N_pix * pixel_variance  (Poisson noise absorbed into variance)
    n_pix    = float(aperture.area)
    flux_err = np.sqrt(n_pix * bkg_var) if bkg_var > 0 else np.nan
    snr      = raw_flux / flux_err if (flux_err > 0 and np.isfinite(flux_err)) else 0.0

    result.update(
        flux_counts=raw_flux,flux_err=flux_err,
        snr=snr,flag_nondet=(snr < SN_THRESHOLD),
    )
    return result


# ── NUV AB magnitude conversion ─────────────────────────────────────────────

def counts_to_nuv_ab(
    flux_counts: float,
    flux_err: float,
) -> tuple[float, float]:
    """
    Convert GALEX NUV count-rate to AB magnitude.

    AB mag = -2.5 * log10(CPS) + 20.08   (standard GALEX NUV zero-point)
    """
    cps     = flux_counts / exptime
    cps_err = flux_err    / exptime

    if cps <= 0:
        return np.nan, np.nan

    mag     = -2.5 * np.log10(cps) + GALEX_NUV_ZP_AB
    mag_err = (2.5 / np.log(10)) * (cps_err / cps)
    return mag, mag_err


def calc_SFR_NUV(NUV_mag, NUV_mag_err, dist_mpc, internal_ext=0.9, internal_ext_err=0.2):
    """
    By Marla Geha 2023
    Convert NUV magnitudes into a SFR
    Based on Iglesias-Paramo (2006), Eq 3
    https://ui.adsabs.harvard.edu/abs/2006ApJS..164...38I/abstract
    """

    # DISTANCE OF HOST (in cm)
    dist = dist_mpc * 3.086e24
    dmod = np.log10(4.0 * np.pi * dist * dist)

    # CORRECT FOR INTERNAL EXTINCTION (assumed to be external extinction corrected)
    m_nuv_ab = NUV_mag - internal_ext

    # CONVERT GALEX m_AB TO FLUX:  erg sec-1 cm-2 Angstrom-1)
    # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
    log_flux_nuv = -0.4 * (m_nuv_ab - 20.08 - 2.5 * np.log10(2.06e-16))

    # LUMINOSITY (erg/s/A-1)
    # 796A is NUV filter width
    log_L_nuv = log_flux_nuv + dmod + np.log10(796)

    # CONVERT TO SOLAR LUMINOSITY
    l_nuv_msun = log_L_nuv - np.log10(3.826e33)

    # CONVVERT TO SFR: EQ 3, inglesias- paramo 2006, also account for Salpeter -> Koupa IMF
    log_SFR_NUV = l_nuv_msun - 9.33 + np.log10(_IMF_FACTOR)

    # PROPAGATE ERRORS: assume ANUV_ERR and measurement errors
    # ANUV_ERR is determined to be consistent with BD_ERR
    log_SFR_NUV_err = 0.4 * np.hypot(NUV_mag_err, internal_ext_err)

    return log_SFR_NUV, log_SFR_NUV_err


# ── Main pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run the full GALEX NUV photometry pipeline.

    Parameters
    ----------
    catalog_path : Path to input catalog (FITS or CSV)
    
    Returns
    -------
    DataFrame of photometry results
    """
    from io import BytesIO  # local import used by download helper

    # Patch BytesIO into outer scope for the download function
    import builtins
    builtins.__dict__["BytesIO"] = BytesIO

    log.info("Loading catalog: v1saga_dwarfs.csv")
    catalog = pd.read_csv('v1saga_dwarfs.csv')

    catalog['zdist'] = cosmo.comoving_distance(catalog['zspec']).value
    
    n_src   = len(catalog)
    log.info("Catalog loaded: %d sources", n_src)

    brick_list,ind_list = get_brick_mapping(catalog)
    dwarf_bricks,brick_mem = np.unique(brick_list,return_inverse=True)

    results = []

    #for idx, row in catalog.iterrows():
    #    pos  = [float(row["ra"]), float(row["dec"])]
    #    log.info("[%d/%d]  RA=%.4f  Dec=%.4f", idx + 1, n_src, pos[0], pos[1])

    Nbk = len(dwarf_bricks)
    log.info("Bricks in TRACTOR: %d",Nbk)

    for i in range(5):  
            if i%100==0: print(i)
            brickname = dwarf_bricks[i]
            dw_mem = np.where(brick_mem==i)[0]

            flag = True
            try: 
                dat = fits.open(sc.get('ls_dr10://south/tractor/' + brickname[:3] + '/tractor-' + brickname + '.fits', mode='fileobj'))
            except:  
                dat = fits.open(sc.get('ls_dr9://north/tractor/' + brickname[:3] + '/tractor-' + brickname + '.fits', mode='fileobj'))
                flag = False

            tract = pd.DataFrame({'objid':dat[1].data['objid'].byteswap().newbyteorder()})

            keys = ['ra', 'dec', 'brickname', 'ebv','flux_r','shape_r','shape_e1','shape_e2']
            for key in keys:
                tract[key] = dat[1].data[key].byteswap().newbyteorder()

            tract['m_r0'] = -2.5*np.log10(tract['flux_r']) + 22.5 - tract['ebv'] * 2.165
            tract['ellip']= np.sqrt(np.square(tract['shape_e1'])+np.square(tract['shape_e2']))
            tract['ba'] =  (1 - tract['ellip'])/(1 + tract['ellip'])
            tract['PA'] = 0.5*np.arctan(tract['shape_e2']/tract['shape_e1'])

            for k in dw_mem: 
                row = catalog.iloc[k]
                pos = [row['ra'],row['dec']]

                bright = tract[(tract['m_r0'] < MASK_R_MAG_LIMIT)&
                               (tract['ra']>(pos[0]-CUTOUT_HALF))&(tract['ra']<(pos[0]+CUTOUT_HALF))&
                               (tract['dec']>(pos[1]-CUTOUT_HALF))&(tract['dec']<(pos[1]+CUTOUT_HALF))]

                # ── 1. Download GALEX cutout ─────────────────────────────────────────
                im_data, wcs = download_galex_cutout(pos)

                base_record = dict(ind=int(row['ind']),
                    ra=pos[0], dec=pos[1],zspec=row["zspec"],r_eff=row["reff"],)

                if im_data is None or wcs is None:
                    log.warning("  No GALEX data for source %d", k)
                    results.append({
                        **base_record,
                        "flux_counts": np.nan, "flux_err": np.nan,
                        "snr": np.nan, "nuv_mag": np.nan, "nuv_mag_err": np.nan,
                        "aperture_type": "none", "a_arcsec": np.nan,
                        "bkg_median": np.nan, "bkg_var": np.nan,
                        "flag_nondet": False, "flag_nodata": True,
                    })
                    continue

                # ── 2. Build background mask ─────────────────────────────────────────
                mask = build_source_mask(im_data, wcs, bright)

                # ── 3. Background estimation ─────────────────────────────────────────
                bkg_median, bkg_var = estimate_background(im_data, mask)
                log.debug("  Bkg median=%.4f  var=%.4f", bkg_median, bkg_var)

                # ── 4. Aperture photometry ───────────────────────────────────────────
                phot = do_aperture_photometry(
                    im_data, wcs,pos=pos,r_eff=bright["shape_r"],
                    ellipticity= 1-bright["ba"],position_angle=bright["PA"],
                    r_mag=bright["ellip"],bkg_median=bkg_median,bkg_var=bkg_var)

                # ── 5. AB magnitude ──────────────────────────────────────────────────
                nuv_mag, nuv_mag_err = counts_to_nuv_ab(phot["flux_counts"], phot["flux_err"],row["zdist"])

                log.info(
                    "  flux=%.3e  SNR=%.1f  mag=%.2f  nondet=%s  nodata=%s",
                    phot["flux_counts"], phot["snr"], nuv_mag if np.isfinite(nuv_mag) else -99,
                    phot["flag_nondet"], phot["flag_nodata"],
                )

                results.append({
                    **base_record,
                    **phot,
                    "nuv_mag": nuv_mag, "nuv_mag_err": nuv_mag_err,
                    "bkg_median": bkg_median,"bkg_var": bkg_var,
                })

    # ── Save ─────────────────────────────────────────────────────────────────
    df  = pd.DataFrame(results)
    df.to_csv('v1GALEX_dwarfs.csv', index=False)
    log.info("Results written to %s", 'v1GALEX_dwarfs.csv')

