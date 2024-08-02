git clone https://github.com/vibansal/HapCUT2.git
cd HapCUT2
git clone https://github.com/samtools/htslib.git
cd htslib
git submodule update --init --recursive
autoreconf -i
./configure --disable-bz2 --disable-lzma
make  

# Run make again

cd ..
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/htslib"
make

# "make" may throw an erorr which can be resolved by commenting out -lcurl -llzma -lbz2 in the extractHairs line (line 20) of MakeFile.
# Run "make" in the HapCUT2 folder again.
# In the event that there are import erorrs, run the aforementioned PATH command again.
