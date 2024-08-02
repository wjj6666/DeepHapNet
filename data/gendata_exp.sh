#!/bin/bash

BWA="bwa-0.7.17/bwa"  # BWA 对齐器的路径
EXTRACT_MATRIX="./ExtractMatrix"  # ExtractMatrix 可执行文件的路径
CAECSEQ="./CAECseq_haplo.py"  # CAECSeq Python 文件的路径
EXTRACTHAIRS="./HapCUT2/build/extractHAIRS"  # HapCUT2 的 extractHAIRS 可执行文件的路径
VERBOSE=false

while getopts f:o:n:v FLAG
do
  case "${FLAG}" in
    f) REF=${OPTARG};;  # 参考基因组的路径
    o) OUTHEAD=${OPTARG};;  # 生成的单倍型文件的基础名称
    n) HAPNUM=${OPTARG};;  # 生成的单倍型数量
    v) VERBOSE=true;;
    *) echo "Invalid command line option: -$FLAG" ;;
  esac
done

# 检查参考基因组文件是否存在
if [[ ! -f $REF ]];then
  echo "Reference genome $REF not found."
  exit 1
fi
echo "Reference genome check completed.........................."

# 如果参考基因组索引不存在，则创建索引
if [[ ! -f "$REF.fai" ]];then
  samtools faidx $REF
  echo "Reference genome indexed.........................."
fi

# 确保输出文件夹存在
HAPFOLDER="./$OUTHEAD"
mkdir -p $HAPFOLDER
echo "Output folder $HAPFOLDER created.........................."

# 删除不必要的索引文件
rm -f "$HAPFOLDER/"*".fai"

# 进行读取对齐
$BWA index $REF
echo "BWA indexing completed.........................."
$BWA mem -t 5 $REF "/home/wangjiaojiao/XHap-master/Deephap/data/SRR6173308/SRR6173308_"*".fastq" | samtools view -h -F 4 > "$HAPFOLDER/$OUTHEAD.sam"
echo "Alignment completed.........................."
samtools sort "$HAPFOLDER/$OUTHEAD.sam" -o "$HAPFOLDER/$OUTHEAD""_sorted.bam"
echo "SAM file sorted.........................."

# 生成读取-SNP矩阵
THRESH=0.1
HAPLEN=$(awk '{print length }' $REF | tail -1)  # 获取参考基因组的长度
#HAPLEN=$(awk '!/^>/{total+=length($0)} END{print total}' $REF)  # 获取整个参考基因组的总长度
echo "Total length of the reference genome: $HAPLEN"

$EXTRACT_MATRIX -f $REF -s "$HAPFOLDER/$OUTHEAD.sam" -t $THRESH -z "$HAPFOLDER/$OUTHEAD" -k $HAPNUM -b 0 -e $HAPLEN -q 0 -l 30 -i 560 
echo "SNP matrix generated.........................."

# 从读取-SNP矩阵创建 VCF 文件
/home/wangjiaojiao/miniconda3/envs/xhap/bin/python snp2vcf.py -r $REF -p "$HAPFOLDER/$OUTHEAD""_SNV_pos.txt" -m "$HAPFOLDER/$OUTHEAD""_SNV_matrix.txt" -o "$HAPFOLDER/$OUTHEAD""_variants.vcf"
echo "VCF file created.........................."

