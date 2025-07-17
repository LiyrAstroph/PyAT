__all__ = ["spec_merge",]

import numpy as np
from pyat import rebin_spectrum, rebin_spectrum_with_error

TINY = 1.0e-50

def spec_merge(wave_list, flux_list, err_list=None):
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

    wave = wave_list[0]
    flux = flux_list[0]
    if err_list is not None:
        err = err_list[0] 
    else:
        err = np.zeros(len(flux))  # Initialize err if not provided

    for i in range(1, len(wave_list)):
        wave_i = wave_list[i]
        flux_i = flux_list[i]

        if err_list is not None:
            err_i = err_list[i]
        else:
            err_i = np.zeros(len(flux_i))  # Initialize err_i if not provided

        overlap_mask_0 = (wave >= wave_i[0]) & (wave <= wave_i[-1])
        overlap_mask_i = (wave_i >= wave[0]) & (wave_i <= wave[-1])

        if not np.any(overlap_mask_0) or not np.any(overlap_mask_i):
            # No overlap, simply append the new spectrum
            wave = np.append(wave, wave_i)
            flux = np.append(flux, flux_i)
            if err_list is not None:
                err = np.append(err, err_i)
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
            flux_0[overlap_mask] = np.average((flux_0[overlap_mask], flux_rebin), axis=0, 
                                               weights=(1/(err_0[overlap_mask]**2+TINY),  1/(err_rebin**2+TINY)))
            err_0[overlap_mask] = np.sqrt(np.average((err_0[overlap_mask]**2, err_rebin**2), axis=0,
                                              weights=(1/(err_0[overlap_mask]**2+TINY), 1/(err_rebin**2+TINY))))

        else:
            flux_rebin = rebin_spectrum(wave_0[overlap_mask], wave_m, flux_m)
            flux_0[overlap_mask] = np.average((flux_0[overlap_mask], flux_rebin), axis=0)


        idx = np.where(~overlap_mask_m)[0]
        wave = np.append(wave_0, wave_m[idx])
        flux = np.append(flux_0, flux_m[idx])
        err = np.append(err_0, err_m[idx])

        arg_sort = np.argsort(wave)
        wave = wave[arg_sort]
        flux = flux[arg_sort]
        err = err[arg_sort]

    if err_list is not None:
        return wave, flux, err
    else:
        return wave, flux