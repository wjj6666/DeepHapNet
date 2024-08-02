import argparse
from multiprocessing import Pool
import numpy as np
from numpy import random as rn
from scipy.spatial.distance import pdist, squareform
from os import path
from copy import deepcopy
from tqdm import tqdm
import time

import logging
from typing import Any, List, Tuple, Optional
from nptyping import NDArray
import torch.optim.lr_scheduler as lr_scheduler
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

from main_real import RetNet_loss, caculate_parameters,assignment,optimise
from embeddings import AE_train

from utils import * 
from retnet.retention import MultiScaleRetention
from retnet.util import ComplexFFN, ComplexGroupNorm, ComplexLayerNorm
from spectralnet._cluster import SpectralNet
from retnet.retnet import RetNet


def train_deephap(SNVdata: SparseSNVMatrixDataset, 
                  gpu: int=-1, 
                  hidden_dim: int = 128, 
                  num_hap: int = 2,
                  num_epoch: int = 2000, 
                  learning_rate: float=1e-4, 
                  beta1: float=1,
                  beta2: float=100,
                  alpha: float=0.1
               ):

    """
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print('The code uses GPU....')
    else:
        device = torch.device('cpu')
        print('The code uses CPU....')
    """
    if gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu))
        else:
            device = torch.device("cpu")
            print("GPUs not available, so CPU being used instead.")
    else:
        device = torch.device("cpu")
    print('DEVICE: ', device)    


        
    # load data
    SNV_matrix = SNVdata.SNV_matrix.todense()
    print('SNP matrix: ', SNV_matrix.shape)
    batch_size = int(np.ceil(len(SNVdata)/5))
    dataloader = DataLoader(SNVdata, batch_size=batch_size, shuffle=True, num_workers=0)    
    

    # Initial read embedding encoder
    savefile="read_AE"
    hidden_dim = 128
    embedAE = AE_train(SNVdata, num_epoch=0, embed_dim=hidden_dim, savefile=savefile).to(device)
    # Initial retnet to extract features
    retnet = RetNet(hidden_dim).to(device)  

    total_params = (sum(p.numel() for p in embedAE.parameters()) + sum(p.numel() for p in retnet.parameters()))
    print('Total number of params: ', total_params)


    retnet_optimizer = optim.AdamW(list(retnet.parameters()) + list(embedAE.parameters()),lr=learning_rate)
    #scheduler = lr_scheduler.CosineAnnealingLR(retnet_optimizer, T_max=num_epoch, eta_min=1e-6)


    mec = []
    mec_min = np.inf

    hap_origin = assignment(SNVdata, embedAE, retnet, num_hap=num_hap, device=device)  # Initial haplotype memberships
    hap_matrix = SNVtoHap(SNV_matrix, hap_origin.cpu().detach().numpy().astype(int), num_hap) 
    mec.append(MEC(SNV_matrix, hap_matrix))
    

    W_over, W_mask, W_dynamic = (x.to(device) for x in caculate_parameters(SNV_matrix))
    MSE = nn.MSELoss()
    for epoch in range(num_epoch):
        retnet_train_loss = 0
        embedAE.train()  
        retnet.train()

        for batch_data, batch_idx in dataloader:
            retnet_optimizer.zero_grad()
            input_data = batch_data.to(device)

            embed, recon = embedAE(input_data)
            AE_loss = MSE(recon,input_data) #+ embedAE.calculate_sparsity_penalty(embed)
            _,Y = retnet(embed[None,:])

            retnet_loss = RetNet_loss(Y[0],
                                      hap_origin[batch_idx],
                                      W_over[batch_idx][:,batch_idx],
                                      W_mask[batch_idx][:,batch_idx],
                                      W_dynamic[batch_idx][:,batch_idx],
                                      beta1,beta2)  + alpha*AE_loss
            retnet_loss.backward()
            retnet_optimizer.step()
            retnet_train_loss += retnet_loss.item()
        retnet_train_loss = retnet_train_loss / len(dataloader)
        #scheduler.step()         

        hap_origin = assignment(SNVdata, embedAE, retnet, num_hap=num_hap,device=device)  
        hap_matrix = SNVtoHap(SNV_matrix, hap_origin.cpu().detach().numpy().astype(int), num_hap)

        mec_curr = MEC(SNV_matrix, hap_matrix)
        if mec_curr <= mec_min:
            mec_min = mec_curr
            hap_matrix_best = 1*hap_matrix
            hap_origin_best = 1*hap_origin
            print('Epoch = %d, MEC = %d' %(epoch, mec_curr))
        mec.append(mec_curr)

        # Display epoch training loss
        with open('Retnet_training_log.txt', 'a') as log_file:
            log_file.write(f"epoch : {epoch + 1}/{num_epoch}, loss = {retnet_train_loss:.2f}\n")
        if epoch % 100 == 0:
            print("epoch : {}/{}, loss = {:.2f}".format(epoch + 1, num_epoch, retnet_train_loss))

    hap_origin_best = hap_origin_best.cpu().numpy()

    best_mec , hap_matrix_best, hap_origin_best = optimise(SNV_matrix,hap_origin_best,num_hap)
    print("The MEC after refine: ",best_mec)   
    return hap_matrix_best, best_mec



def best_match(rec_hap, new_hap):
	if np.shape(rec_hap) != np.shape(new_hap):
		raise ValueError("Input arguments should have the same shape.")
	distance_table = np.zeros((len(rec_hap), len(new_hap)))
	for i, rh in enumerate(rec_hap):       
		for j, nh in enumerate(new_hap):   
			distance_table[i, j] = HM_distance(rh, nh)        
	index = permutations(range(new_hap.shape[0]))
	min_distance = np.inf      
	distance = []             
	for matching in index:     
		count = 0
		for i, match_idx in enumerate(matching):    
			count += distance_table[i, match_idx]
		distance.append(count)
		if count < min_distance:
			best_matching = matching
			min_distance = count
	return best_matching    


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filehead", help="Prefix of required files",type=str, required=True)
    parser.add_argument("-p", "--ploidy", help="Ploidy of organism",default=2, type=int)
    parser.add_argument("-a", "--algo_runs", help="Number of experimental runs per dataset",default=1, type=int)
    parser.add_argument("-g", "--gpu", help='Number of GPUs to run deephap',default=0, type=int)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = parser()
    chunk_size = 250 
    overlap_frac = 0.2 

    datapath = 'data/NA12878/' + args.filehead + '/' +  args.filehead + '_SNV_matrix.txt'
    savepath = 'data/NA12878/' + args.filehead + '/' +  args.filehead + '_deephap_hap_matrix.npz'
    SNV_matrix = read_sparseSNVMatrix(datapath)
    if os.path.exists(savepath):
        hap_matrix = np.load(savepath)['hap']
    else:
        hap_matrix = np.zeros((args.ploidy, SNV_matrix.shape[1]), dtype=int)

    def train_deephap_map(spos, SNVdata, gpu=args.gpu, num_runs=args.algo_runs):      
        mec_min = np.inf             
        hap_matrix_best = np.zeros((args.ploidy, chunk_size), dtype=int)     
        for r in range(num_runs):    

            hap_matrix_run, mec_run = train_deephap(SNVdata, 
                                                    gpu=gpu,
                                                    hidden_dim=128,
                                                    num_hap=args.ploidy, 
                                                    num_epoch=2000)
            if mec_run < mec_min:
                  mec_min = mec_run
                  hap_matrix_best = hap_matrix_run
        return spos, hap_matrix_best         
    

    SNV_matrix_list = chunk_data(datapath, chunk_size=chunk_size, overlap_frac=overlap_frac)          
    #gpu_list = range(0, 6, 1) 
    gpu_list = [3,4,5]      
    
    print(args.gpu)
    print(gpu_list)
    recon_end = 0  
    pool = Pool(processes=args.gpu)     
    print('Created process pool')       

    start_time = time.time()

    for d in range(0, len(SNV_matrix_list), args.gpu):
        chunk_starts, chunk_SNVdata = zip(*SNV_matrix_list[d:d + args.gpu]) 
        print('Running deephap on chunks starting at: ', chunk_starts)
        res_d = pool.starmap(train_deephap_map, zip(chunk_starts, chunk_SNVdata, gpu_list))      
        print('Finished running deephap on chunks starting at: ', chunk_starts)
        # Stitch haplotype chunks together
        pos_list, hap_chunk_list = zip(*res_d)      
        for pos, hap_chunk in zip(pos_list, hap_chunk_list):
            np.savez('data/NA12878/' + args.filehead + '/hap_chunk/'+ 'hap_pos_' + str(pos), hap_chunk)      

            print('Chunk starts')
            print('MEC: ', MEC(SNV_matrix[:, pos:pos+hap_chunk.shape[1]].toarray(), hap_chunk))  
            # print('CPR: ', compute_cpr(hap_chunk, true_hap[:, pos:pos+hap_chunk.shape[1]]))

            if pos == 0:
                hap_matrix[:, :hap_chunk.shape[1]] = hap_chunk
                recon_end = hap_chunk.shape[1]
            else:  # determine best match
                rec_hap = hap_matrix[:, pos:recon_end]
                match = best_match(rec_hap, hap_chunk[:, :recon_end - pos])
                hap_matrix[:, pos:pos + hap_chunk.shape[1]] = hap_chunk[match,:]
                recon_end = pos + hap_chunk.shape[1]
                # print('MATCH: ', match)

        print('Status so far')

        print('MEC: ', MEC(SNV_matrix[:, :recon_end].toarray(), hap_matrix[:, :recon_end]))
        # print('CPR: ', compute_cpr(hap_matrix[:, :recon_end], true_hap[:, :recon_end]))

np.savez(savepath, hap = hap_matrix)   
print('Finished in %d seconds.' %(time.time()-start_time))

