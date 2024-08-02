# File to process SNV_matrix.txt file and remove uninformative reads (covering 0 or 1 SNPs)

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input file containing unfiltered read-SNP matrix")
parser.add_argument("-o", "--output", help="File to store filtered read-SNP matrix")

args = parser.parse_args()

SNV_matrix_raw = np.loadtxt(args.input, dtype=int)
SNV_matrix = SNV_matrix_raw[np.sum(SNV_matrix_raw != 0, axis=1) > 1]
np.savetxt(args.output, SNV_matrix, fmt='%d')

