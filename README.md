# DeepHap: 

## About
DeepHap is a comprehensive framework for learning similarities between reads based on Retentive Network (RETNET), which is the infrastructure for a large-scale language model, and then clustering reads through a deep clustering architecture, eventually accomplishing the task of assembling diploid or polyploid haplotypes.

The current implementation of DeepHap uses Python3, PyTorch. 

## Dependencies
- PyTorch >= 1.10
- Numpy
- C++
- Samtools

Where possible, additional dependencies have been included in the GitHub repository.

## Assumed directory structure
The `data` folder stores the data for all experiments, and all DeepHap scripts are in the current directory.


## Input
The input to the DeepHap model is the `SNV_matrix.txt` file, which needs to be obtained by processing before running the pipeline, the exact script can be found in the `data` folder.

## Output
The best results eventually obtained by the DeePHap pipeline are stored in the following files:  

`haptest_retnet_res.npz`:**.npz** file storing the reconstructed haplotype(rec_hap), the read attribute (rec_hap_origin), and the ground-truth haplotype(true_hap) if applicable.   

`deephap_model`:**State_dict** storing the convolutional(embedAE)and retnet coding layers(retnet) in DeepHap.

## Usage
Run **main_real.py** : 

    python mian_real.py -f soltub_region1 -p 4 -a 5 -g 3

Run **mian_semi.py** ：

    # using simulated short-reads dataset
    python main_semi.py -f 3_soltub_10x -r /home/wangjiaojiao/XHap-master/deephap/data/short_ref/solanum_tuberosum.fa -p 3 -c 10 -n 5 -a 5 -g 3 
    
    # using simulated long-reads dataset
    python main_semi.py -f 3_human_80x -r /home/wangjiaojiao/XHap-master/deephap/data/long_ref/human_sample.fa -p 3 -c 80 -n 5 -a 5 -g 3 --long

Run **mian_semi.py** ：

    python main_sparse.py -f chr21 -p 2 -a 1 -g 3

| Option | Description |
|--------|-------------|
| **-f** | Prefix of required files |
| **-r** | Reference FASTA |
| **-p** | Ploidy of organism |
| **-c** | Coverage |
| **-n** | Number of datasets |
| **-a** | Number of experimental runs per dataset|
| **-g** | GPU to run deephap |
| **--long** | True if using long reads |


## Citation
