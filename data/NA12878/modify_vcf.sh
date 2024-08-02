#!/bin/bash

# Modify VCF file such that the CHROM field reads chr[] instead of [].

while getopts f:o: FLAG
do
  case "${FLAG}" in
    f) FILE=${OPTARG};;  # Input VCF file
    o) OUTFILE=${OPTARG};;  # Output VCF file
    *) echo "Invalid command line option: -$FLAG" ;;
  esac
done

HEADER=$(grep "^#" $FILE | wc -l)
head -n $HEADER $FILE > $OUTFILE  # Copy header to output file
tail -n +$(($HEADER + 1)) $FILE | awk '{print "chr"$0}'  >> $OUTFILE  # Add chr to each vriant line


# while read line; do
#     if [[ $line == \#* ]]; then
#         echo $line >> "$OUTFILE"
#     else
#         echo "chr"$line >> "$OUTFILE"
#     fi
#     done < $FILE