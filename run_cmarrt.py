import sys
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
# each of these is from the peak caller directory
import bedgraph
import cmarrt
import peak

def convert_coordinate(coord, start, res):
    return (coord*res) + start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cmarrt on a bedgraph file')
    parser.add_argument('bedgraph_file', type=str,
                        help='input bedgraph file of continuous data across the genome.')
                                
    parser.add_argument('wi', type=int, 
                        help='half the size of window to do an average over')
    parser.add_argument('chrom_name', type=str, 
                         help='name of the chromosome, must match bedgraph file')
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
                        help='basepair resolution for input data (default = 1 bp). \
                                Values are sampled at specified resolution. \
                                If resolution specified is higher than input data \
                                then values are duplicated over bedgraph ranges. Areas with \
                                no coverage in bedgraph file are replaced with nans')
    parser.add_argument('--nan_cutoff', type=float, default = 0.1,
                        help='fraction of nans acceptable within a window (default = 0.1)')
    parser.add_argument('--sample_frac', type=float, default=0.1, 
                        help='fraction of data to sample for background distro (default = 0.1)')
    parser.add_argument('--consolidate', type=int, default = None, help="number of entries to consolidate peaks over (default=wi)")
    parser.add_argument('--plots', action='store_true', help='plot distributions? (default = False)')
    parser.add_argument('--input_numpy', action='store_true', help='is the input a .npy array instead of a bedgraph file?')
    parser.add_argument('--np_start', type=int, default=None,
                         help='length of the chromosome in 0-based coordinates, needed for numpy input')
    parser.add_argument('--np_end', type=int, default=None,
                         help='length of the chromosome in 0-based coordinates, needed for numpy input')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    in_file = args.bedgraph_file
    wi = args.wi
    percentile = args.p
    out_prefix = args.o
    q_cutoff = args.q
    seed = args.s
    if args.consolidate:
        consolidate = args.consolidate
    else:
        consolidate = wi

    # Read input data

    if args.input_numpy:
        if args.np_start is None or args.np_end is None:
            raise ValueError("--np_start and --np_end must be specified if using --input_numpy ")
        span = (args.np_start, args.np_end)
        array = np.load(args.bedgraph_file)
        array_length = len(array)
        expected_min_size = np.floor((args.np_end - args.np_start)/args.resolution)
        if array_length < expected_min_size:
            raise ValueError("Length of array %s does not match min expected \
                    size %s. Make sure --resolution, --np_start, and --np_end \
                    are set appropiately and that the numpy input array has values \
                    for each step size"%(array_length, min_size))
    else:
        bgfile = bedgraph.BedGraphFile(in_file)
        bgfile.load_whole_file()
        span, array = bgfile.to_array(args.chrom_name)
        if args.resolution > 1:
            array = array[::args.resolution]

    # Run CMARRT
    arrays, peaks, fig = cmarrt.main(array, wi=wi, 
                                     percentile=percentile, sample_frac=args.sample_frac, 
                                     nan_cutoff=args.nan_cutoff,
                                     q_cutoff= q_cutoff, seed = 0,
                                     consolidate = consolidate, plots=args.plots)
    score = arrays[0]
    pvalues = arrays[1]
    qvalues = arrays[2]

    # Make peaks
    all_peaks = peak.PeakList()
    for i, entry in enumerate(peaks):
        start, end = entry
        these_invals = array[start:end]
        these_pvalues = pvalues[start:end]
        these_qvalues = qvalues[start:end]
        true_start = convert_coordinate(start, span[0], args.resolution)
        true_end = convert_coordinate(end, span[0], args.resolution)
        try:
            max_idx = np.nanargmax(these_invals)

            this_peak = peak.Peak(chrm = args.chrom_name, start=true_start, end=true_end, 
                              name = "%s_%s"%(out_prefix,i), 
                              score = int(these_qvalues[max_idx]*10),
                              signalval = these_invals[max_idx], 
                              pval = these_pvalues[max_idx], 
                              qval = these_qvalues[max_idx], peak = max_idx)
            all_peaks.add_Peak(this_peak)
        except:
            max_idx = np.argmax(these_qvalues)
            print(true_start, true_end, "failed", qvalues[max_idx], pvalues[max_idx])
    # Write peaks
    all_peaks.write_narrowPeak_file(out_prefix + ".narrowPeak")
                  
    if args.plots:    
        plt.tight_layout()
        fig.savefig(out_prefix + ".png") 
