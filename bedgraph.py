""" bedgraph

A set of classes to handle and stream bedgraph files

Implemented:
    BedGraphLine - a class to store the data from a single bedgraph line
                   allows for iteration through the range that a line covers
                   for single basepair value streaming
TO DO:
    Random access of a bedgraph file
"""
import gzip
import numpy as np

class BedGraphFile(object):

    def __init__(self, filename):
        self.chroms = chroms
        self.filename = filename
        if self.filename.endswith(".gz"):
            self.open_func = gzip.open
        else:
            self.open_func = open

    def iter_file_by_line(self):
        with self.open_func(self.filename) as f:
            for line in f:
                yield line

    def load_whole_file(self):
        self.data = {}
        self.chroms = []
        this_chrom = None
        for raw_line in self.iter_file_by_line():
            line = BedGraphLine.from_line(raw_line)
            if line.chrom != this_chrom:
                self.data[line.chrom] = BedGraphChrom(line.chrom)
                chrom_data = self.data[line.chrom]
                self.chroms.append(line.chrom)
            else:
                chrom_data.add_line(line)

    def to_array(self, chrm):
        this_chrm = self.data[chrm]
        span = (this_chrm.start, this_chrm.end)
        array = this_chrm.to_array()
        return (span, array)

    def pull_region(self, chrm, start, stop):
        """ Warning this will spawn a child process to access the file, this
        also requires bgzipped and tabixed files. If doing this over and over
        again in parallel, consider reading in chunks of the file and accessing
        that way
        """
        import tabix
        region = tabix.tabix_pull(self.filename, chrm, start, stop)
        array = np.full(stop-start, np.nan)
        for line in region:
            line = BedGraphLine.from_line(line)
            array[line.start:line.end] = line.value
        return array


class BedGraphChrom(object):

    def __init__(self, chrom=None, lines=None):
        self.lines = lines
        self.chrom = chrom
        self.start = None
        self.end = None

    def __iter__(self):
        for line in self.lines:
            yield line

    def add_line(self, line):
        self.lines.append(line)

    def to_array(self):
        self.start = self.lines[0].start
        self.end = self.lines[-1].end
        array = np.full(self.end-self.start, np.nan)
        for line in self:
            array[line.start-self.start:line.end-self.start] = line.value
        return array

class BedGraphLine(object):

    @classmethod
    def from_line(cls, line):
        linearr = line.strip().split('\t')
        return cls(chrom=linearr[0], start=int(linearr[1]), end=int(linearr[2]),
                   value=float(linearr[3]))

    def __init__(self, chrom = None, start = None, end = None, value = None):
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)
        self.value = float(value)

    def __iter__(self):
        for val in range(self.start, self.end):
            yield self.value

    def __str__(self):
        return "%s\t%s\t%s\t%s"%(self.chrom,self.start,self.end,self.value)

if __name__ == "__main__":
    import sys
    
    infile = sys.argv[1]
    outfile = sys.argv[2]
    chrm = sys.argv[3]

    bg_file = BedGraphFile(infile)
    bg_file.load_whole_file()
    span, out_array = bg_file.to_array(chrm)
    print("Span of output array for chrm %s: %s - %s"%(chrm, span[0], span[1]))
    np.save(outfile, out_arr)

