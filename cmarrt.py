from __future__ import division
"""
Python implementation of the CMARRT algorithm (Kuan et al. 2008)

Written by Michael Wolfe
2017-04-02

"""
from scipy import stats
import numpy as np
import numpy.ma as ma
import matplotlib
# gets matplotlib to stop trying to use x11
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import warnings


def autocor(array, max_lag, mean = None):
    """ Calculate the autocorrelation of an array up for values of k up to
    the max lag.

    The autocorrelation is defined as the following:

    rho(k) = sum_t=1^T-k (Y_t - Y^bar)*(Y_t+k - Y^bar)
             -----------------------------------------
                    sum_t=1^T (Y_t -Y^bar)^2
    Where:
            T is the total number of values in the window
            Y^bar is the mean of the entire array (all windows)
            k is the lag 

    Args: 
        array (np.array)- array to calculate the autocorrelation for
        max_lag (int) - maximum value of lag (k) to calculate the 
                        autocorrelation for
        mean (float) - mean of the array if not already subtracted from the
                       array (if running on many samples it is faster to pass
                       in an array that the mean has already been subtracted
                       from)
    Returns:
        autocor (np.array) - an array containing the autocorrelation at each
                             value of k. I.e. autocor[1] is the autocorrelation
                             at a k of 1.
    """
    # if a mean is supplied, then subtract it, otherwise assume that mean has
    # already been subtracted from the array
    if mean:
        norm_array = array - mean
    else:
        norm_array = array
    # calculate the denominator of the equation
    var = np.nansum(np.square(norm_array))
    # initialize a vector to hold the autocorrelation values for each k
    autocor = np.zeros(max_lag+1)
    # calculate the autocorrelation for each k
    for i in range(max_lag+1):
        # calculate the numerator by shifting the array by k but keeping within
        # the window
        num = np.nansum(norm_array[:norm_array.size-i]*norm_array[i:])
        autocor[i] = num/var
    return autocor


def autocor_over_regions(array, gaps, max_lag, min_cont):
    """ Calculate the autocorrelation as the weighted average of the
    autocorrelation over continous regions across the array as long as the
    regions are as long as the min_cont variable. 
    
    The average is weighted by the length of each region. Anywhere with a nan is
    not considered a continous region. The top 5% of data should be masked as
    nans before going into this function

    Args:
        array (np.array) - array over the entire chromosome, with masked
        gaps (np.array) - boolean array containing masked locations
        max_lag (int) - maximum lag to calculate the autocorrelation for
        min_cont (int) - minimum size of a continous region to be considered for
                         autocorrelation calculation.

    Returns:
        autocor (np.array) - an array containing the autocorrelation at each
                             value of k.
    """
    # create lists to store the autocorrelations and region sizes for each
    # region
    autocor_regions = []
    region_size = []
    bad_vars = []
    start = 0
    # Loop through all the locations of the nans
    for loc in np.where(gaps)[0]:
        # take a view from the start to the first nan
        view = array[start:loc]
        start = loc + 1
        # if the view isn't big enough skip it
        if view.size < min_cont:
            continue
        else:
            # if a region is all the same value then you will get a variance
            # of zero, in order to deal with this we will ignore these, record
            # how many there are and what the average size windows are
            autocor_tmp = autocor(view, max_lag, mean=np.mean(view))
            if not np.all(np.isfinite(autocor_tmp)):
                bad_vars.append(view.size)
            else:
                # otherwise we calculate the autocorrelation for the view
                autocor_regions.append(autocor_tmp)
                region_size.append(view.size)
    # next we figure out how big all the regions are for a weighted average
    weights = np.asarray(region_size)
    # log how many regions the autocorrelation was estimated from
    logging.info("Autocorrelation estimated from %s regions of average length: %s"\
                  %(weights.size, np.mean(weights)))
    if len(bad_vars) > 0:
        logging.warning("Ignored %s windows of average size %s with all the same value"%(len(bad_vars), np.mean(bad_vars)))
    total_weight = np.sum(weights)
    weights = weights/total_weight
    # multiply each autocorrelation by its weight
    autocor_regions = [region*weight for region, weight in zip(autocor_regions, weights)]
    autocor_regions = np.asarray(autocor_regions)
    # sum to get the weighted average of autocorrelation over all regions
    autocor_regions = np.sum(autocor_regions, 0)
    return(autocor_regions)

def estimate_covar(autocor_vec, sample_var, mask = None):
    """ Estimate the covariance term for CMARRT

    The covariance term is defined as follows:

    cov(Yj, Yj+k) = rho(k)*sigma^2
    
    sum_j=i-wi^i+wi sum_k!=j cov(Yj, Yk)

    Args:
        autocor_vec (np.array) - output from the autocor_over_regions function
        sample_var (float) - the estimated sample variance for the entire 
                             chromosome.
    Returns:
        covar_term (float) - the estimated covariance term
    """
    # take off the first value of the autocorrelation since we dont consider
    # the case where k = j, also scale it by the sample variance

    covar_vals = autocor_vec*sample_var
    window_size = covar_vals.size
    if mask is None:
        mask = np.zeros(window_size, dtype=bool)
    covar_sum = 0
    for i in range(window_size):
        for j in range(window_size):
            if i == j or mask[i] or mask[j]:
                continue
            else:
                covar_sum += covar_vals[abs(i-j)]
    return(covar_sum)

def get_percentile_cutoff(array, percent):
    """ Returns the value that the top percent % of data is above

    Args:
        array (np.array) - array of total data, handles nan and infs
        percent(float) - percentile
    Returns:
        cutoff (float) - datapoint where the top percent of data is over
    """
    return np.percentile(array[np.isfinite(array)], 100-percent)

def standardized_moving_average(array, wi, mean, sample_var, autocor_vec,
                                nan_thresh = 0.1):
    """ Calculate the standardized moving average for an entire array

    The standardized moving average statistic (Si) is defined as the following:

    Ti = (1/(2*wi+1))*sum_j=i-wi^i+wi(Yi)
    Var(Ti) = (1/(2*wi+1)^2)*((2*wi+1)*sigma^2 + est_cov_term)
    Si = Ti/sqrt(var(Ti))

    Args:
        array (np.array) - chromosome array with all data
        wi (int) - size of one half of the window
        mean (float) - the mean to subtract from each window
        sample_var (float) - estimated variance
        cov (float) - estimated covariance term for a window
    Returns:
        np.array the size of the chromosome
    """
    # pad the array so that the beginning and end values can be calculated
    padded_array = np.concatenate((array[array.size-wi:], array, array[:wi]))
    padded_mask = ~np.isfinite(padded_array)
    # make the output array the size of the original array
    output_array = np.zeros(array.size)
    # calculate the size of the window
    window_size = 2*wi+1
    # estimate covariance for a full window
    cov = estimate_covar(autocor_vec, sample_var)
    logging.info("Estimated covariance for %s bp window: %s"%(window_size, cov))
    # calculate the variance with the correlation structure for a full window
    var_Ti = np.sqrt(sample_var/window_size + cov/np.square(window_size))
    logging.info("Standard Deviation for %s bp window: %s"%(window_size,var_Ti))
    # calculate the minimum size of a window thats acceptable:
    nan_thresh = int(round((1-nan_thresh)*window_size))
    logging.info("Throwing out windows < %s bps"%nan_thresh)
    nan_windows = 0
    skipped_windows = 0
    for i in range(array.size):
        view = padded_array[i:i+window_size]
        mask = padded_mask[i:i+window_size]
        if np.any(mask):
            new_view = view[~mask]
            if new_view.size < nan_thresh:
                output_array[i] = np.nan
                skipped_windows += 1
                continue
            else:
                cov = estimate_covar(autocor_vec, sample_var, mask)
                var_view = np.sqrt(sample_var/new_view.size + 
                                   cov/np.square(new_view.size))
                output_array[i] = (np.mean(new_view) - mean)/var_view
                nan_windows += 1
        else:
            output_array[i] = (np.mean(view)-mean)/var_Ti
    logging.info("Windows skipped %s"%skipped_windows)
    logging.info("Windows with nans in them %s"%nan_windows)
    return output_array

def calculate_pvalues(array):
    mask = np.isnan(array)
    new_array = np.zeros(array.size)
    new_array[mask] = np.nan
    new_array[~mask] = -1*stats.norm.logsf(array[~mask])/np.log(10)
    return new_array

def neg_logp_to_neg_logq(array):
    """ Convert -log_10(pvalues) to -log_10(qvalues) using the benjamini-hochberg
    procedure. This gives answers identical to R's p.adjust (if it dealt with log
    transformed p values)

    Be careful with large arrays. Two additional array.size are 
    created internally, one is returned.

    Args:
        array (np.array) - contains -log_10(pvalues), unsorted.
    Returns:
        qvalues (np.array) - -log_10(qvalues) in the same order as p values
    """
    # get an index for the sorted array, This is sorted from largest p value
    # to smallest (since -log(p) higher means smaller p)
    idx = np.argsort(array)
    asize = array.size
    qvalues = np.zeros(asize)
    M = np.log10(asize)
    qval_prev = 0
    for i, idx_val in enumerate(idx):
        qval = array[idx_val] + (np.log10(asize-i) - M)
        qval = max(qval, qval_prev)
#        print idx_val,array[idx_val], np.log(asize-i), M, qval_this,qval,np.e**(-1*qval)
        qvalues[idx_val] = qval
        qval_prev = qval
    return qvalues

def call_peaks(array, consolidate = 0):
    """ Take a logical array and find the start and stop of ones across
    the array. Consolidate any peaks that are within consolidate.

    TODO: Deal with peaks that go over the end of the array

    Args:
        array (np.array) - logical array to call peaks from.
        consolidate (int) - number of bins to consolidate peaks over
    Returns:
        peak_indices (list of lists) - a list of [start, stop] indices in the
                                        array
    """
    # first find all the places where there is a change from 1 to 0 or vice
    # versa, here we pad the incoming array with a zero on each end, then take
    # the difference along the array and finally take the absolute value to flip
    # the places where the difference was 0 - 1
    changes = np.abs(np.diff(np.concatenate(([0], array.view(np.int8), [0]))))
    # changes is now a logical array, I then find all the indices where changes
    # happen and reshape them into an ndarray of start, stop locations
    start_stops = np.where(changes)[0].reshape(-1, 2)
    if start_stops.size == 0:
        logging.warning("No bp was considered to be within a peak.")
        return []

    # now lets consolidate any peaks that are within consolidate
    consolidate_peaks = [[start_stops[0][0], start_stops[0][1]]]
    consolidated_peaks = 0
    removed_peaks = 0
    for start, stop in start_stops[1:]:
        if start - consolidate_peaks[-1][1] < consolidate:
            consolidate_peaks[-1][1] = stop
            consolidated_peaks += 1
        else:
            if stop-start > consolidate:
                consolidate_peaks.append([start, stop])
            else:
                removed_peaks += 1
    logging.info("Consolidated %s peaks within %s bps"%(consolidated_peaks, consolidate))
    logging.info("Removed %s peaks < %s bps"%(removed_peaks, consolidate))
    return consolidate_peaks

def estimate_background(array, sample_frac = .10, res = 512j, axes = None, bins=10):
    """ Use a kernel density estimate to determine the background sd and
    mean for a 1 dimensional array of ChIP-seq data.

    Args:
        array (np.array) - array of data, cannot contain nans or inf
        sample_frac (float) - fraction of data to sample for background
        res (complex) - number of data points for gaussian estimate. Default is
                        equivalent to R's density function
        axes (matplotlib.axes) - an axis to plot data onto
        bins (int) - number of bins for plotted histograms

    Returns:
       center (float) - the center value of the background distribution
       variance (float) - the variance of the background distribution

    """
    # calculate how much to sample
    datapoints = int(round(array.size*sample_frac))
    # compute the density
    data_min = array.min()
    data_max = array.max()
    x_vals = np.ogrid[data_min:data_max:res]
    logging.info("Estimating Density from a sample containing %s of %s datapoints"%(datapoints, array.size))
    estimate = stats.gaussian_kde(np.random.choice(array, datapoints))
    y_vals = estimate(x_vals)

    # find the center of the data
    center_index = y_vals.argmax()
    center = x_vals[center_index]
    
    # estimate the background variance from the lower half of the data
    left_index,  = np.where(y_vals > 0.001)
    lower_lim = x_vals[left_index[0]]
    logging.info("Creating Background Distribution")
    lower_half = array[(array >= lower_lim) & (array <= center)]
    background = np.concatenate([lower_half, 2*center-lower_half])
    variance = np.var(background)
    if axes:
        logging.info("Plotting Distributions")
        hist_bins = np.linspace(data_min, data_max, num=bins)
        axes.hist(array, bins=hist_bins, color='blue', alpha = 0.5, label='Raw Data')
        axes.hist(background, bins=hist_bins, color='green', alpha = 0.5,label='Background')
        axes.set_title("Background (g) All Data (b)")
#        axes.set_xlabel("Value")
#        axes.set_ylabel("Frequency")
#        axes.legend()
    return(center, variance)

def main(array, wi = 100, percentile = 5, sample_frac = 0.10, nan_cutoff = 0.10,
         q_cutoff = 0.001, seed = None, min_auto_size = 500, consolidate = None, 
         plots = False):
    """ Main function for CMARRT. Calls peaks from a 1-d numpy array of continous
    data across a genomic region.

    TODO - deal with circular regions

    Args:
        array (np.array) - array containing data, must be continuous across the
                           genomic region of interest. NaNs ands infs are dealt
                           with internally
        wi (int)         - half the size of the window to calculate the 
                           standardized moving average over. Window size is then
                           2*wi + 1
        percentile (int) - what upper percent of the data to exclude from the 
                           autocorrelation calculation. I.e the top (1-percentile)
                           data is included
        sample_frac (float) - percentage of data to use for background calculation
        min_auto_size (int) - minimum size of a region to be considered for 
                              autocorrelation calculation. Must be larger than
                              2*wi+1
        seed     (int)   - random seed to set for sampling of the data to 
                           determine the background distribution.
        q_cutoff (float) - qvalue cutoff to be consider significant and part of
                           a peak
        consolidate (int) - peaks that are within this many basepairs will be
                            consolidated into one peak. Default is wi
        plots (boolean)   - produce diagnostic plots?
    Returns:
        [stat, pvalue, qvalue] - list of arrays the length of the input array 
                                 with the standardized moving average stat,
                                 the pvalue and the qvalue
        peaks  - numpy array with start stop locations of peaks
        fig    - if plots, returns the figure object containing the plots
                 otherwise this is None.
    """
    ## READ IN USER PARAMS ##
    # if plotting is happening, create a subplot for all the plots
    if plots:
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    # otherwise set each of these variable to None, so the plots aren't created
    else:
        fig, ax1, ax2, ax3 = (None,)*4
    # if a seed was used, alert the user to which random seed was used
    if seed is not None:
        logging.info("Using random seed %s"%seed)
        np.random.seed(seed)
    # Default number of bp to consolidate over to half the window size
    if consolidate is None:
        consolidate = wi
    # conver the qvalue cutoff to -log10(qvalue)
    q_cutoff = -1*np.log10(q_cutoff)

    ## FIND THE AUTOCORRELATION WITHIN A WINDOW
    # find where nans and inf are in the array
    missing_data = ~np.isfinite(array)
    # find the top percentile of the data when the inf and nans are removed
    cut_off = get_percentile_cutoff(array[~missing_data], percentile)
    logging.info("%s %% percentile cutoff value: %s"%(percentile, cut_off))
    # find the locations where the data is above this percentile to mask it
    with warnings.catch_warnings():
        # ignore warning for nans in the comparison
        warnings.simplefilter("ignore")
        masked_data = array > cut_off
    # find the autocorrelation over the continous regions
    # passing in the gaps as both the masked and missing data
    logging.info("Finding autocorrelation from regions of size > %s"%(min_auto_size))
    autocor = autocor_over_regions(array, (masked_data | missing_data), wi*2, 
                                   min_auto_size)
    # plot this autocorrelation for each bp in a window size
    if ax1:
        ax1.bar(np.arange(1, autocor.size+1), autocor, linewidth=0, color = 'black')
        ax1.set_xlim([0, autocor.size+1])
        ax1.set_ylim([0, 1])
        ax1.set_title("Autocorrelation")

    ## ESTIMATE BACKGROUND VARIANCE AND CENTER
    center, var = estimate_background(array[~missing_data], 
                                      sample_frac = sample_frac,
                                      axes = ax2, bins = 50)
    logging.info("Total Variance: %s"%np.var(array[~missing_data]))
    logging.info("Background Variance: %s"%var)
    logging.info("Total Mean: %s"%np.mean(array[~missing_data]))
    logging.info("Background Center: %s"%center)

    ## CALCULATE THE STANDARDIZED MOVING AVERAGE
    logging.info("Calculating Standardized Moving Average")
    stat = standardized_moving_average(array, wi, center, var, autocor, 
                                       nan_thresh = nan_cutoff)

    ## CALCULATE P AND Q VALUES
    logging.info("Calculating p-values")
    pvalues = calculate_pvalues(stat)
    logging.info("Calculating q-values")
    mask = np.isnan(pvalues)
    qvalues = np.zeros(pvalues.size)
    qvalues[mask] = np.nan
    qvalues[~mask] = neg_logp_to_neg_logq(pvalues[~mask])
    if ax3:
        hist_bins = np.linspace(0, 1, 50)
        ax3.hist(10**(-1*pvalues[~np.isnan(pvalues)]), hist_bins, color = 'blue', 
                 alpha = 0.5, label = "p-value")
        ax3.hist(10**(-1*qvalues[~np.isnan(qvalues)]), hist_bins, color = 'red', 
                 alpha = 0.5, label = "q-value")
        ax3.set_title("p-val (b) q-val (r)")

    ## CALL PEAKS
    logging.info("Calling Peaks with -log10(qvalue) > %s "%q_cutoff)
    with warnings.catch_warnings():
        # ignore warning for nans in the comparison
        warnings.simplefilter("ignore")
        peaks = call_peaks(qvalues > q_cutoff, consolidate = consolidate)
    logging.info("%s Peaks found"%len(peaks))

    return ([stat, pvalues, qvalues], peaks, fig)
