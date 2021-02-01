import sys
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pyBigWig
import cmarrt
import peak

def convert_coordinate(coord, start, res):
    return (coord*res) + start

def bigwig_to_arrays(bw, res = None):
    """
    Convert a bigwig to a dictionary of numpy arrays, one entry per contig
    
    Args:
        bw - pyBigWig object
        res - resolution that you want the array in
    Returns:
        arrays (dict) - a dictionary of numpy arrays
    """
    arrays = {}
    for chrm in bw.chroms().keys():
        arrays[chrm] = contig_to_array(bw, chrm, res)
    return arrays

def contig_to_array(bw, chrm, res = None):
    """
    Convert single basepair bigwig information to a numpy array
    

    Args:
        bw - a pyBigWig object
        chrm - name of chromosome you want 
        res - resolution you want data at in bp. 

    Returns:
        outarray - numpyarray at specified resolution

    """
    chrm_length = bw.chroms(chrm)
    # makes an array at 1 bp resolution
    out_array = bw.values(chrm, 0, chrm_length, numpy=True)
    if res:
        out_array = out_array[::res]
    return out_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cmarrt on a BigWig file.\
            Each contig is run individually and the results are compiled.')
    parser.add_argument('bigwig_file', type=str,
                        help='input bedgraph file of continuous data across the genome.')
                                
    parser.add_argument('wi', type=int, 
                        help='half the size of window to do an average over')
    parser.add_argument('-p', type=float, default= 5,
                        help='top p%% of data that peaks are expected to fall in (default = 5)')
    parser.add_argument('-o', type=str,
                        help='prefix including full path to output files (default = CMARRT)',
                        default='CMARRT')
    parser.add_argument('-q', type=float, default=0.001,
                        help='qvalue cutoff for peak calling (default = 0.001)')
    parser.add_argument('-s', type=int, default = 0,
                        help='random seed for background distribution generation (default = 0)')
    parser.add_argument('--resolution', type=int, default = 1,
                        help='basepair resolution for input data (default = 1 bp).')
    parser.add_argument('--nan_cutoff', type=float, default = 0.1,
                        help='fraction of nans acceptable within a window (default = 0.1)')
    parser.add_argument('--sample_frac', type=float, default=0.1, 
                        help='fraction of data to sample for background distro (default = 0.1)')
    parser.add_argument('--consolidate', type=int, default = None, help="number of entries to consolidate peaks over (default=wi)")
    parser.add_argument('--plots', action='store_true', help='plot distributions? (default = False)')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    in_file = args.bigwig_file
    wi = args.wi
    percentile = args.p
    out_prefix = args.o
    q_cutoff = args.q
    seed = args.s
    if args.consolidate:
        consolidate = args.consolidate
    else:
        consolidate = wi

    # read in file 
    inf = pyBigWig.open(args.bigwig_file)
    
    # convert to a dictionary of arrays
    arrays = bigwig_to_arrays(inf, res = args.resolution)
    inf.close()

    # Make peak list

    # Make peaks
    all_peaks = peak.PeakList()

    for contig, array in arrays.items():
        logging.info("Running CMARRT on contig %s"%(contig))
        # Run CMARRT
        arrays, peaks, fig = cmarrt.main(array, wi=wi, 
                                         percentile=percentile, sample_frac=args.sample_frac, 
                                         nan_cutoff=args.nan_cutoff,
                                         q_cutoff= q_cutoff, seed = 0,
                                         consolidate = consolidate, plots=args.plots)
        score = arrays[0]
        pvalues = arrays[1]
        qvalues = arrays[2]
    
        for i, entry in enumerate(peaks):
            start, end = entry
            these_invals = array[start:end]
            these_pvalues = pvalues[start:end]
            these_qvalues = qvalues[start:end]
            true_start = convert_coordinate(start, 0, args.resolution)
            true_end = convert_coordinate(end, 0, args.resolution)
            try:
                max_idx = np.nanargmax(these_invals)
    
                this_peak = peak.Peak(chrm = contig, start=true_start, end=true_end, 
                                  name = "%s_%s"%(out_prefix,i), 
                                  score = int(these_qvalues[max_idx]*10),
                                  signalval = these_invals[max_idx], 
                                  pval = these_pvalues[max_idx], 
                                  qval = these_qvalues[max_idx], peak = max_idx)
                all_peaks.add_Peak(this_peak)
            except:
                max_idx = np.argmax(these_qvalues)
                logging.warning(true_start, true_end, "failed", qvalues[max_idx], pvalues[max_idx])

            if args.plots:    
                plt.tight_layout()
                plt.suptitle(contig)
                plt.savefig(out_prefix + "_%s.png"%(contig))
    # Write peaks
    all_peaks.write_narrowPeak_file(out_prefix + ".narrowPeak")
