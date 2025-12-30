#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["spec_merge",]

import numpy as np
from pyat import rebin_spectrum, rebin_spectrum_with_error

TINY = 1.0e-50

def spec_merge(wave_list, flux_list, err_list=None, scale=False, SN_limit=None):
    """
    Merge multiple spectra into a single spectrum.

    Parameters
    ----------
    wave_list : list of 1D array-like
        Wavelength arrays for each spectrum.
    flux_list : list of 1D array-like
        Flux arrays for each spectrum.
    err_list : list of 1D array-like, optional
        Error arrays for each spectrum. If provided, will be used to compute the merged error.
    scale : bool, optional
        If True, the merged spectrum will be scaled to match the flux of the individual spectra.
    SN_limit: float, optional
        Signal-to-noise ratio limit for including individual spectral pixel in the merged spectrum.

    Returns
    -------
    merged_wave : 1D array
        Merged wavelength array.
    merged_flux : 1D array
        Merged flux array.
    merged_err : 1D array, optional
        Merged error array if `err_list` is provided.
    """
    
    # Ensure all input lists are of the same length
    if len(wave_list) != len(flux_list):
        raise ValueError("wave_list and flux_list must have the same length.")
    
    if err_list is not None and len(err_list) != len(wave_list):
        raise ValueError("If provided, err_list must have the same length as wave_list and flux_list.")
    
    if SN_limit is not None and err_list is None:
        raise ValueError("err_list must be provided if SN_limit is set.")
    
    # first sort the wave_list
    wave_sort = np.zeros(len(wave_list))
    for i in range(len(wave_list)):
        if len(wave_list[i]) != 0:
            wave_sort[i] = wave_list[i][0]
        else:
            wave_sort[i] = np.inf
    arg_sort = np.argsort(wave_sort)
    wave_list = [wave_list[i] for i in arg_sort]
    flux_list = [flux_list[i] for i in arg_sort]
    err_list = [err_list[i] for i in arg_sort] if err_list is not None else None

    i_start = 1
    for i in range(len(wave_list)):
        if len(wave_list[i]) != 0:
            if SN_limit is None:
                wave = wave_list[i]
                flux = flux_list[i]
                if err_list is not None:
                    err = err_list[i]
                else:
                    err = np.zeros(len(flux))  # Initialize err if not provided
            else:
                sn = flux_list[i] / (err_list[i] + TINY)
                mask = (sn >= SN_limit) & (err_list[i] > 0)
                wave = wave_list[i][mask]
                flux = flux_list[i][mask]
                err = err_list[i][mask]

            i_start = i+1
            break
        i_start += 1
    
    if i_start == len(wave_list) + 1:
        raise ValueError("All input spectra are empty.")

    for i in range(i_start, len(wave_list)):

        if len(wave_list[i]) == 0:
            continue
        
        if SN_limit is None:
            wave_i = wave_list[i]
            flux_i = flux_list[i]

            if err_list is not None:
                err_i = err_list[i]
            else:
                err_i = np.zeros(len(flux_i))  # Initialize err_i if not provided
        else:
            sn = flux_list[i] / (err_list[i] + TINY)
            mask = (sn >= SN_limit) & (err_list[i] > 0)
            wave_i = wave_list[i][mask]
            flux_i = flux_list[i][mask]
            err_i = err_list[i][mask]

        overlap_mask_0 = (wave >= wave_i[0]) & (wave <= wave_i[-1])
        overlap_mask_i = (wave_i >= wave[0]) & (wave_i <= wave[-1])

        if not np.any(overlap_mask_0) or not np.any(overlap_mask_i):
            # No overlap, simply append the new spectrum
            wave = np.append(wave, wave_i)
            flux = np.append(flux, flux_i)
            if err_list is not None:
                err = np.append(err, err_i)
            
            arg_sort = np.argsort(wave)
            wave = wave[arg_sort]
            flux = flux[arg_sort]
            err = err[arg_sort]

            continue

        # check which spectrum is is finer 
        dw_0 = (wave[overlap_mask_0][-1] - wave[overlap_mask_0][0])/len(overlap_mask_0)
        dw_i = (wave_i[overlap_mask_i][-1] - wave_i[overlap_mask_i][0])/len(overlap_mask_i)

        if dw_i < dw_0:
            # wave_i is finer, rebin wave
            overlap_mask = overlap_mask_i
            overlap_mask_m = overlap_mask_0
            wave_0 = wave_i 
            flux_0 = flux_i 
            err_0 = err_i 

            wave_m = wave 
            flux_m = flux
            err_m = err
        else:
            # wave is finer, rebin wave_i
            overlap_mask = overlap_mask_0
            overlap_mask_m = overlap_mask_i
            wave_0 = wave 
            flux_0 = flux 
            err_0 = err 

            wave_m = wave_i 
            flux_m = flux_i
            err_m = err_i
            
        
        if err_list is not None:
            flux_rebin, err_rebin = rebin_spectrum_with_error(wave_0[overlap_mask], wave_m, flux_m, err_m)
            if scale:
                sum0 = np.average(flux_0[overlap_mask], weights=1.0/(err_0[overlap_mask]**2+TINY))
                sum1 = np.average(flux_rebin, weights=1.0/(err_rebin**2+TINY))
                factor = sum0/sum1
            else:
                factor = 1.0

            flux_0[overlap_mask] = np.average((flux_0[overlap_mask], factor*flux_rebin), axis=0, 
                                               weights=(1/(err_0[overlap_mask]**2+TINY),  1/(factor**2*err_rebin**2+TINY)))
            err_0[overlap_mask] = np.sqrt(np.average((err_0[overlap_mask]**2, factor**2*err_rebin**2), axis=0,
                                              weights=(1/(err_0[overlap_mask]**2+TINY), 1/(factor**2*err_rebin**2+TINY))))

        else:
            flux_rebin = rebin_spectrum(wave_0[overlap_mask], wave_m, flux_m)
            if scale:
                sum0 = np.average(flux_0[overlap_mask])
                sum1 = np.average(flux_rebin)
                factor = sum0/sum1
            else:
                factor = 1.0
            flux_0[overlap_mask] = np.average((flux_0[overlap_mask], factor*flux_rebin), axis=0)

        idx = np.where(~overlap_mask_m)[0]
        wave = np.append(wave_0, wave_m[idx])
        flux = np.append(flux_0, factor*flux_m[idx])
        err = np.append(err_0, factor*err_m[idx])

        arg_sort = np.argsort(wave)
        wave = wave[arg_sort]
        flux = flux[arg_sort]
        err = err[arg_sort]

    if err_list is not None:
        return wave, flux, err
    else:
        return wave, flux