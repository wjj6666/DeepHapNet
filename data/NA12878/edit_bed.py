input_path = "/home/wangjiaojiao/dataset/NA12878_HG001_GRCh37/HG001_GRCh37_1_22_v4.2.1_callablemultinter_gt0.bed"
output_path = "/home/wangjiaojiao/XHap-master/A/data/NA12878/chr21/chr21.bed"

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        if line.startswith('21\t'):
            line = 'chr' + line
            outfile.write(line)
