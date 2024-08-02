from numpy import random as rn
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fasta", help="FASTA file",
                    type=pathlib.Path, required=True)
parser.add_argument("-t", "--truth", help="Storing indel",
                    type=pathlib.Path)
parser.add_argument("-i", "--insertion", help="Insertion length",
                    type=int, default=None)
parser.add_argument("-d", "--deletion", help="Deletion length",
                    type=int, default=None)
args = parser.parse_args()

FASTA_file = args.fasta
ins_len = args.insertion
del_len = args.deletion
indel_file = args.truth

# FASTA_file = "haptest/combined.fa"
# ins_len = None
# del_len = 100

if (ins_len is None) == (del_len is None):
    raise ValueError('Only one of insertion or deletion length must be non-zero.')

hap_list = []
header_list = []
with open(FASTA_file, "r") as f:
    while True:
        line = f.readline().strip()
        if not line:
            break
        elif line[0] == '>':  # Header string
            header_list.append(line)
        else:  # Genome string
            hap_list.append(line)

# Add insertion or deletion to second haplotype
hap_len = len(hap_list[0])
if ins_len:  # Insertion
    ins_start = rn.randint(1000, hap_len - 1000)
    ins_seq = ''.join(rn.choice(['A', 'C', 'G', 'T'], size=ins_len))
    hap = hap_list[-1]
    hap_list[-1] = hap[:ins_start] + ins_seq + hap[ins_start:]
else:  # Deletion
    if del_len > hap_len:
        raise ValueError('Haplotype cannot support deletion of length %d.' %del_len)
    del_start = rn.randint(1000, hap_len - 1000)
    hap = hap_list[-1]
    hap_list[-1] = hap[:del_start] + hap[del_start+del_len:]
    
# Write modified haplotypes back to file
with open(FASTA_file, "w") as f:
    for head, hap in zip(header_list, hap_list):
        f.write(head)
        f.write('\n')
        f.write(hap)
        f.write('\n')

# Store indel ground truth
with open(indel_file, "w") as f:
    if ins_len:
        f.write('START POS: %d' %(ins_start))
        f.write('\n')
        f.write('INSERTED SEQ: %s' %ins_seq)
        f.write('\n')
    else:    
        f.write('START POS: %d' %(del_start))
        f.write('\n')
