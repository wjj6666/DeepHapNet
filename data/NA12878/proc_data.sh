#!/bin/bash

REGION_FILE="chr21.bed"

# awk '{print "chr"$0}' > $REGION_FILE

for i in {21..22}
do
    CHR="chr"$i
    BAM=$CHR"_reads.bam"
    FILT=$CHR"_reg_reads.sam"

if [[ ! -f "$BAM.bai" ]];then
    echo "$BAM.bai not found. Indexing..."
    samtools index $BAM
fi

samtools view -h -L $REGION_FILE $BAM > $FILT


echo "$CHR done"

done
