# Read SNPs from VCF file and write heterozygous SNPs to a new VCF file

import numpy as np
from os import path
import argparse
from tqdm import tqdm

def read_vcf(fname, outpre):

    if not path.isfile(fname):
        raise OSError("VCF file (to be read) not found.")

    vcf_out = outpre + '.vcf'
    snp_out = outpre + '_SNV_pos.txt'
    hap_out = outpre + '_true_haplotypes.txt'

    f_out = open(vcf_out, 'w')

    snp_pos = []  # 1-indexed SNP positions
    haps = ['', '']
    base_set = set(['A', 'C', 'G', 'T'])

    with tqdm(total=path.getsize(fname)) as pbar:
        with open(fname) as f:
            for line in f:
                pbar.update(len(line))
                if line.startswith("#"):  # Keep header
                    f_out.write(line)
                    # continue
                else:
                    line_split = line.strip().split()
                    ref_allele = line_split[3]
                    alt_allele = line_split[4]
                    if (ref_allele not in base_set
                        or alt_allele not in base_set):  # Ignore indels
                        continue

                    format_fields = line_split[8].split(':')
                    gt_idx = format_fields.index('GT')
                    gt_val = line_split[9].split(':')[gt_idx]
                    # print(format_fields, gt_val)

                    if '|' in gt_val:  # phasing info present
                        gt = gt_val.split('|')
                        if gt[0] != gt[1]:  # heterozygous SNP
                            f_out.write(line)
                            snp_pos.append(line_split[1])
                            haps[0] = haps[0] + (ref_allele if gt[0] == '0' else alt_allele)
                            haps[1] = haps[1] + (ref_allele if gt[1] == '0' else alt_allele)
                        else:  # Ignore homozgyous SNPs
                            continue
                    else:  # No phasing info. Not storing this in ground truth.
                        continue

    
    # Writing SNP_positions info
    f_snp = open(snp_out, 'w')
    f_snp.write(' '.join(snp_pos))
    f_snp.close()

    f_hap = open(hap_out, 'w')
    for i, hap in enumerate(haps):
        f_hap.write('>hap' + str(i) + '\n')
        f_hap.write(hap + '\n')
    f_hap.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse SNPs from VCF file')
    parser.add_argument('-f', '--file', required=True, help='VCF file')
    parser.add_argument('-o', '--out', required=True, help='Output file prefix')
    args = parser.parse_args()

    read_vcf(args.file, args.out)

