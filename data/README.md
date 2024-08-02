# Data simulation

## 1. Semi-experimental data
#### 1.1 Need tools:   
- [HaploSim](https://github.com/EhsanMotazedi/Haplosim)    
- [ART](https://www.niehs.nih.gov/research/resources/software/biostatistics/art/index.cfm)   
- [PBSIM2](https://github.com/yukiteruono/pbsim2)   
- BWA-MEM

#### 1.2 Semi-experimental data generation comprises the following steps:
- Generating haplotypes from selected reference genome using _haplotypegenerator.py_ from **HaploSim**.
- Generating short or long reads using **ART** or **PBSIM2** respectively.
- Aligning reads to reference genome using **BWA-MEM**.
- Variant calling and generating read-SNP matrix using _ExtractMatrix_ (or _ExtractMatrix_longread_ for long reads).

#### 1.3 Simulated data can be obtained through the bash script **gendata_semiexp.bash**. Below are the parameters for running this script:
| Option | Description |
|--------|-------------|
| **-f** | Path to reference genome |
| **-o** | Output directory |
| **-n** | Ploidy |
| **-c** | Coverage per haplotype, i.e., coverage/n|
| -i | Lenght of insertion to be placed in one of the generated haplotype sequences |
| -d | Lenght of deletion to be placed in one of the generated haplotype sequences |
| -v | Added if more verbose output is desired (useful for debugging)|


## 2. Experimental data
#### 2.1 The Bash script **gendata_exp.bash** can be used to process real potato data (stored in the `SolTub` folder).
Run **gendata_exp.bash** example：

    ./gendata_exp.sh -f /deephap/data/SolTub/region_reference/region_1.fasta -o soltub_region1 -n 4
    
#### 2.2 Included reference genomes
- **solanum_tuberosum.fa:** 10 kbp sample of Solanum Tuberosum Chromosome 5 genome (used for semi-experimental data generation),stored in the `short_ref` folder.
- **human_sample.fa:**: 100 kbp sample of GrCh38 genome (used for semi-experimental data generation),stored in the `long_ref` folder.
- **region_[*].fa:** 10 kbp samples of Solanum Tuberosum Chromosome 5 genome (used for experimental data processings),stored in the `SolTub` folder.