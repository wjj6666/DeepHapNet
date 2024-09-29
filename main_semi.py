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
    parser.add_argument("-g", "--gpu", help='GPU to run deephap',default=-1, type=int)
    parser.add_argument("--long", help="True if using long reads",action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = parser()
    mec_expt = []
    swer_expt = []
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
        swer = []
        for r in range(args.algo_runs):
            # Train deephap on generated data
            print('RUN %d for %s' %(r+1, fhead))
            mec_r,swer_r = train_deephapnet(fhead,
                                  num_epoch=20,
                                  gpu=args.gpu, 
                                  num_hap=args.ploidy,
                                  check_swer=True)
            if len(mec) == 0 or mec_r < min(mec):
                shutil.copy('data/' + fhead + '/haptest_retnet_res.npz', 'data/' + fhead + '/haptest_retnet_res_best.npz')
                shutil.copy('data/' + fhead + '/deephap_model', 'data/' + fhead + '/deephap_model_best')
            mec.append(mec_r)
            swer.append(swer_r)
        print('MEC scores for deephap: ', mec)
        print('SWER scores for deephap: ', swer)
        
        r_best = np.argmin(mec)
        print('Best MEC: %.3f, Corresponding SWER: %.3f' %(mec[r_best], swer[r_best]))
        mec_expt.append(mec[r_best])
        swer_expt.append(swer[r_best])

    print('Average MEC: %.3f +/- %.3f' %(np.mean(mec_expt), np.std(mec_expt)))
    print('Average SWER: %.3f +/- %.3f' %(np.mean(swer_expt), np.std(swer_expt)))
