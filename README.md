# Python implementation of the CMARRT algorithm #

This algorithm was originally developed for analyzing ChIP-chip data:

Kuan PF, Chun H, Keles S. CMARRT: A Tool for the Analysis of ChIP-chip
Data from Tiling Arrays by Incorporating the Correlation Structure.
Pac Symp Biocomput. 2008;515–526. 

Here, I have adapted this algorithm for use on any vector of
continuous data (i.e. log2(extracted/ input) ChIP-seq data). The
algorithm currently will only work on data for one chromosome at a time.

# Summary of usage and parameters #
```
usage: run_cmarrt.py [-h] [-p P] [-o O] [-q Q] [-s S]
                     [--resolution RESOLUTION] [--nan_cutoff NAN_CUTOFF]
                     [--sample_frac SAMPLE_FRAC] [--consolidate CONSOLIDATE]
                     [--plots] [--input_numpy] [--np_start NP_START]
                     [--np_end NP_END]
                     bedgraph_file wi chrom_name

Run cmarrt on a bedgraph file

positional arguments:
  bedgraph_file         input bedgraph file of continuous data across the
                        genome.
  wi                    half the size of window to do an average over
  chrom_name            name of the chromosome, must match bedgraph file

optional arguments:
  -h, --help            show this help message and exit
  -p P                  top p% of data that peaks are expected to fall in
                        (default = 5)
  -o O                  prefix including full path to output files (default =
                        CMARRT)
  -q Q                  qvalue cutoff for peak calling (default = 0.001)
  -s S                  random seed for background distribution generation
                        (default = 0)
  --resolution RESOLUTION
                        basepair resolution for input data (default = 1 bp).
                        Values are sampled at specified resolution. If
                        resolution specified is higher than input data then
                        values are duplicated over bedgraph ranges. Areas with
                        no coverage in bedgraph file are replaced with nans
  --nan_cutoff NAN_CUTOFF
                        fraction of nans acceptable within a window (default =
                        0.1)
  --sample_frac SAMPLE_FRAC
                        fraction of data to sample for background distro
                        (default = 0.1)
  --consolidate CONSOLIDATE
                        number of entries to consolidate peaks over
                        (default=wi)
  --plots               plot distributions? (default = False)
  --input_numpy         is the input a .npy array instead of a bedgraph file?
  --np_start NP_START   length of the chromosome in 0-based coordinates,
                        needed for numpy input
  --np_end NP_END       length of the chromosome in 0-based coordinates,
                        needed for numpy input
```
