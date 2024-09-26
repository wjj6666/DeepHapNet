import numpy as np
import subprocess
import os
import shutil
import argparse
from main_real import train_deephapnet

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filehead", help="Prefix of required files",type=str, required=True)
    parser.add_argument("-r", "--reference", help="Reference FASTA",type=str, required=True)
    parser.add_argument("-p", "--ploidy", help="Ploidy of organism",default=2, type=int)
    parser.add_argument("-c", "--cov", help="Coverage",default=10, type=float)
    parser.add_argument("-n", "--num_expt", help="Number of datasets",default=1, type=int)
    parser.add_argument("-a", "--algo_runs", help="Number of experimental runs per dataset",default=1, type=int)
    parser.add_argument("-g", "--gpu", help='GPU to run DeepHapNet',default=-1, type=int)
    parser.add_argument("--long", help="True if using long reads",action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = parser()
    mec_expt = []
    for i in range(args.num_expt):
        fhead = args.filehead + "_iter" + str(i+1)  
        # Generate data
        cov = np.round(args.cov/args.ploidy, 3) 
        os.chdir('data')
        if not args.long:  # Short reads
            subprocess.run(['bash', 'gendata_semiexp.bash', '-f', args.reference, '-o', fhead, '-n', str(args.ploidy), '-c', str(cov)])
        else:  # Long reads
            subprocess.run(['bash', 'gendata_longread_semiexp.bash', '-f', args.reference, '-o', fhead, '-n', str(args.ploidy), '-c', str(cov)])
        os.chdir('../')

        mec = []
        best_mec = float('inf')
        for r in range(args.algo_runs):
            # Train DeepHapNet on generated data
            print('RUN %d for %s' %(r+1, fhead))
            mec_r = train_deephapnet(fhead,
                                  num_epoch=2000,
                                  gpu=args.gpu, 
                                  num_hap=args.ploidy)
            if len(mec) == 0 or mec_r < min(mec):
                best_mec = mec_r
                shutil.copy('data/' + fhead + '/deephapnet.npz', 'data/' + fhead + '/deephapnet_best.npz')
                shutil.copy('data/' + fhead + '/deephapnet_model', 'data/' + fhead + '/deephapnet_model_best')
            mec.append(mec_r)
        print('MEC scores for DeepHapNet: ', mec)
        print('Best MEC: %d' % best_mec)
        mec_expt.append(best_mec)
    print('Average MEC: %.3f +/- %.3f' %(np.mean(mec_expt), np.std(mec_expt)))
