#!/bin/bash

BWA="bwa-0.7.17/bwa"  
EXTRACT_MATRIX="./ExtractMatrix" 
CAECSEQ="./CAECseq_haplo.py"  
EXTRACTHAIRS="./HapCUT2/build/extractHAIRS"  
VERBOSE=false

while getopts f:o:n:v FLAG
do
  case "${FLAG}" in
    f) REF=${OPTARG};;  
    o) OUTHEAD=${OPTARG};;  
    n) HAPNUM=${OPTARG};;  
    v) VERBOSE=true;;
    *) echo "Invalid command line option: -$FLAG" ;;
  esac
done

if [[ ! -f $REF ]];then
  echo "Reference genome $REF not found."
  exit 1
fi
echo "Reference genome check completed.........................."

if [[ ! -f "$REF.fai" ]];then
  samtools faidx $REF
  echo "Reference genome indexed.........................."
fi

HAPFOLDER="./$OUTHEAD"
mkdir -p $HAPFOLDER
echo "Output folder $HAPFOLDER created.........................."

rm -f "$HAPFOLDER/"*".fai"

$BWA index $REF
echo "BWA indexing completed.........................."
$BWA mem -t 5 $REF "./data/SRR6173308/SRR6173308_"*".fastq" | samtools view -h -F 4 > "$HAPFOLDER/$OUTHEAD.sam"
echo "Alignment completed.........................."
samtools sort "$HAPFOLDER/$OUTHEAD.sam" -o "$HAPFOLDER/$OUTHEAD""_sorted.bam"
echo "SAM file sorted.........................."

THRESH=0.1
HAPLEN=$(awk '{print length }' $REF | tail -1)
#HAPLEN=$(awk '!/^>/{total+=length($0)} END{print total}' $REF)  
echo "Total length of the reference genome: $HAPLEN"

$EXTRACT_MATRIX -f $REF -s "$HAPFOLDER/$OUTHEAD.sam" -t $THRESH -z "$HAPFOLDER/$OUTHEAD" -k $HAPNUM -b 0 -e $HAPLEN -q 0 -l 30 -i 560 
echo "SNP matrix generated.........................."


python snp2vcf.py -r $REF -p "$HAPFOLDER/$OUTHEAD""_SNV_pos.txt" -m "$HAPFOLDER/$OUTHEAD""_SNV_matrix.txt" -o "$HAPFOLDER/$OUTHEAD""_variants.vcf"
echo "VCF file created.........................."

