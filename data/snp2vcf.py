# File to generate VCF file from read-SNP matrix

import numpy as np
from os import path
import argparse

def gen_vcf(ref_file: str, pos_file: str,
            read_SNP_matfile: str, out_file: str) -> None:
    
    """
    Create a VCF file based on input read-SNP matrix and SNP positions.
    
    Parameters
    -----------
    ref_file: str
        Reference FASTA file
    pos_file: str
        Text file containing SNP positions (whitespace separator)
    read_SNP_matfile: str
        File containing read-SNP matrix (integer-valued)
    out_file:
        Name of output VCF file to be created (should end with ".vcf")
    
    """

    if not path.isfile(ref_file):
            raise OSError("Reference file not found.") 
    if not path.isfile(read_SNP_matfile):
            raise OSError("Read-SNP matrix file not found.")
    if not path.isfile(pos_file):
        raise OSError('SNP position file not found.')
    if path.splitext(out_file)[-1] != ".vcf":
        return ValueError("Output file name should end in '.vcf'")

    base2int = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 0}  # mapping of base to int
    int2base = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: '-'}  # mapping of base to int 
    
    # Figure out alleles at each SNP
    SNV_matrix = np.loadtxt(read_SNP_matfile, dtype=int)
    allele_list = []
    for j in range(np.shape(SNV_matrix)[1]):
        alleles, allele_counts = np.unique(SNV_matrix[:, j], return_counts=True)
        if alleles[0] == 0:
            alleles = alleles[1:]
            allele_counts = allele_counts[1:]
        if len(allele_list) > 2:
            alleles = alleles[np.argsort(allele_counts)[-2:]]
        allele_list.append(alleles)

    # Read SNP positions
    with open(pos_file, "r") as f:
        pos_str = f.readline().split()
        pos = np.array([int(ps) - 1 for ps in pos_str]) # Convert to int, assuming pos_file is 1-indexed
    
    # Read reference FASTA
    with open(ref_file, "r") as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            elif line[0] == '>':  # ignore header strings
                refname = line.strip().split()[0][1:]  # chromosome name
            else:
                hap_ref = np.array([line[p] for p in pos])


    with open(out_file, "w") as f:
        f.write('\t'.join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]) + '\n')
        
        for i, alleles_int in enumerate(allele_list):
            alleles = np.fromiter(map(lambda a: int2base[a], alleles_int), dtype='U1')
            if hap_ref[i] in alleles:
                gt_list = [str(j) for j, a in enumerate(alleles)]
            else:  # Reference not present in reads
                gt_list = [str(j + 1) for j, a in enumerate(alleles)]

            allele_alt = alleles[alleles != hap_ref[i]]

            gt_str = '/'.join(gt_list)

            # Output line in VCF file is chromose, position, ID, 'referene base', 'alternate base', 'quality',
            # 'filter, 'INFO', 'FORMAT', 'Sample' 
            outline_list = [refname, str(pos[i] + 1), "snp" + str(i+1), hap_ref[i], ','.join(allele_alt), '.',
                            'PASS', '*', 'GT:GQ', gt_str + ':100']
#         print('\t'.join(outline_list))
            f.write('\t'.join(outline_list) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate VCF file from reference FASTA and read-SNP matrix.')
    parser.add_argument('-r', '--ref', required=True, help='Reference FASTA file')
    parser.add_argument('-p', '--pos', required=True, help='Whitespace-separated SNP position file')
    parser.add_argument('-m', '--mat', required=True, help='Integer-valued read-SNP matrix')
    parser.add_argument('-o', '--out', required=True, help='Output file')

    args = parser.parse_args()
    gen_vcf(args.ref, args.pos, args.mat, args.out)
