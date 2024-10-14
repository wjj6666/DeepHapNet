import argparse
import os
import shutil
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils import MEC, SNVtoHap, SNVMatrixDataset,save_ckp,read_true_hap,SWER
from embeddings import ReadAE, AE_train

from spectralnet._cluster import SpectralNet
from retnet.retnet import RetNet

def setting_seed(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def optimise(SNV_matrix, hap_origin, num_hap):
    initial_temp = 0.5 
    final_temp = 0.1  
    alpha = 0.7  
    current_temp = initial_temp

    labels = set(range(num_hap))
    reads = SNV_matrix.shape[0]

    current_hap_matrix = SNVtoHap(SNV_matrix, hap_origin, num_hap)
    best_mec = MEC(SNV_matrix, current_hap_matrix)

    best_origin = hap_origin.copy()
    
    global_best_mec = best_mec  
    global_best_origin = best_origin.copy()  


    while current_temp > final_temp:
        for node in range(reads):
            current_label = hap_origin[node]
            for label in labels:
                if label != current_label:
                    temp_hap_origin = best_origin.copy()
                    temp_hap_origin[node] = label
                    current_hap_matrix = SNVtoHap(SNV_matrix, temp_hap_origin, num_hap)
                    mec = MEC(SNV_matrix, current_hap_matrix)
                    if mec < best_mec or np.random.rand() < np.exp((best_mec - mec) / current_temp):
                        best_mec = mec
                        best_origin = temp_hap_origin.copy()
                        if mec < global_best_mec:
                            global_best_mec = mec
                            global_best_origin = temp_hap_origin.copy()

        current_temp *= alpha  

    global_best_hap_matrix = SNVtoHap(SNV_matrix, global_best_origin, num_hap)
    return global_best_mec, global_best_hap_matrix, global_best_origin


def caculate_parameters(SNV_matrix):
    num_reads = np.shape(SNV_matrix)[0]
    W_sim = np.zeros((num_reads, num_reads))
    W_dissim = np.zeros((num_reads, num_reads))
    W_mask = np.zeros((num_reads, num_reads), dtype=bool)
    W_dynamic = np.zeros((num_reads, num_reads))

    read_lengths = np.sum(SNV_matrix != 0, axis=1)
    for i, read_i in enumerate(tqdm(SNV_matrix)):
        len_i = read_lengths[i]  
        for j, read_j in enumerate(SNV_matrix):
            len_j = read_lengths[j]  
            overlap = (read_i != 0) & (read_j != 0)
            if np.any(overlap):  
                W_mask[i, j] = True
                W_sim[i, j] = np.sum((read_i == read_j)[(read_i != 0) & (read_j != 0)])
                W_dissim[i, j] = np.sum((read_i != read_j)[(read_i != 0) & (read_j != 0)])
                W_dynamic[i, j] = np.sum(overlap) * (max(len_i,len_j))

    W_over = (W_sim-W_dissim)/(W_sim + W_dissim + 1e-10)
    np.fill_diagonal(W_over, 1.)
    W_sim = torch.from_numpy(W_sim)
    W_dissim = torch.from_numpy(W_dissim)
    W_over = torch.from_numpy(W_over)
    W_mask = torch.from_numpy(W_mask)
    W_dynamic = torch.from_numpy(W_dynamic)

    return  W_over, W_mask, W_dynamic


def assignment(SNVdataset: SNVMatrixDataset, 
               ae: ReadAE, 
               retnet: RetNet, 
               device: torch.cuda.device=torch.device("cuda"), 
               num_hap: int=2):

    dataloader_full = DataLoader(SNVdataset, batch_size=len(SNVdataset), num_workers=0)
    for _, (data, idx) in enumerate(dataloader_full):
        SNV_onehot = data.to(device)
        
    ae.eval() 
    retnet.eval()  
    
    embed, _ = ae(SNV_onehot)  
    features,_ = retnet(embed[None,:])
    features = features.detach()
    features = features[0]

    siamese_hiddens = [128,128,64,num_hap]
    spectral_hiddens = [128, 128, 64, num_hap]  
    spectralnet = SpectralNet(n_clusters=num_hap, should_use_siamese=True,is_sparse_graph=False, siamese_hiddens=siamese_hiddens,spectral_hiddens=spectral_hiddens,spectral_epochs =2,device=device)
    spectralnet.fit(features) 
    labels = spectralnet.predict(features)
    return labels

def RetNet_loss(retnet_output: torch.Tensor, 
                origin: torch.Tensor,
                Wover: torch.Tensor, 
                Wmask: torch.Tensor, 
                Wdynamic: torch.Tensor,
                beta1: float = 0.1,
                beta2: float = 0.1):

    origin_onehot = F.one_hot(origin + 1,num_classes=5).float()
    y = torch.matmul(origin_onehot, origin_onehot.transpose(0,1))
    obj_reg =  torch.sum(Wdynamic * Wmask * (retnet_output - Wover)**2)

    # Contrastive loss
    pos_loss = (1/(2*retnet_output.shape[0])) * y * torch.pow(1-retnet_output, 2)
    neg_loss = (1/(2*retnet_output.shape[0])) * (1 - y) * torch.pow(torch.clamp(retnet_output - beta1, min=0.0), 2)
    contrastive_loss = Wdynamic * (pos_loss + neg_loss)
    contrastive_loss = torch.sum(contrastive_loss)
    Loss = contrastive_loss + beta2*obj_reg #beta1 * triplet_loss +  beta2 * obj_reg  + beta3 * loss_contrastive 

    return Loss


def train_deephapnet(outhead: str, 
                  hidden_dim: int = 128, 
                  num_hap: int = 2, 
                  num_epoch: int = 2000, 
                  gpu: int=2,
                  check_swer:bool = False,
                  learning_rate: float=1e-4,
                  beta1: float=1,
                  beta2: float=100,
                  lamda: float=0.1):
 
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU....')
    else:
        device = torch.device('cpu')
        print('The code uses CPU....')

    datapath = 'data/' + outhead + '/' + outhead + '_SNV_matrix.txt'
    gt_file = 'data/' + outhead + '/combined.fa'
    pos_file = 'data/' + outhead + '/' + outhead + '_SNV_pos.txt'
    if check_swer:
        true_haplo = read_true_hap(gt_file, pos_file)  

    SNVdata = SNVMatrixDataset(datapath)
    SNV_matrix = np.loadtxt(datapath, dtype=int)
    SNV_matrix = SNV_matrix[np.sum(SNV_matrix != 0, axis=1) > 1] 
    print('SNP matrix: ', SNV_matrix.shape)
    batch_size = int(np.ceil(len(SNVdata)/5))
    hidden_dim = 128 
    dataloader = DataLoader(SNVdata, batch_size=batch_size,shuffle=True, num_workers=0)    
    
    savefile="read_AE"
    embedAE = AE_train(SNVdata, num_epoch=0, embed_dim=hidden_dim, savefile=savefile).to(device)
    retnet = RetNet(hidden_dim).to(device)  

    mec = []
    mec_min = np.inf

    # Initial haplotype memberships
    hap_origin = assignment(SNVdata, embedAE, retnet, num_hap=num_hap, device=device)  
    hap_matrix = SNVtoHap(SNV_matrix, hap_origin.cpu().detach().numpy().astype(int), num_hap) 

    mec.append(MEC(SNV_matrix, hap_matrix))

    W_over, W_mask, W_dynamic = (x.to(device) for x in caculate_parameters(SNV_matrix))

    MSE = nn.MSELoss()
    retnet_savefile = 'data/' + outhead + '/deephapnet_ckp'
    retnet_optimizer = optim.AdamW(list(retnet.parameters()) + list(embedAE.parameters()),lr=learning_rate)

    for epoch in range(num_epoch):
        retnet_train_loss = 0
        embedAE.train()  
        retnet.train()

        for batch_data, batch_idx in dataloader:
            retnet_optimizer.zero_grad()
            input_data = batch_data.to(device)
            embed, recon = embedAE(input_data)
            AE_loss = MSE(recon,input_data) 
            _,Y = retnet(embed[None,:])

            retnet_loss = RetNet_loss(Y[0],
                                      hap_origin[batch_idx],
                                      W_over[batch_idx][:,batch_idx],
                                      W_mask[batch_idx][:,batch_idx],
                                      W_dynamic[batch_idx][:,batch_idx],
                                      beta1,beta2)  + lamda*AE_loss
            retnet_loss.backward()
            retnet_optimizer.step()
            retnet_train_loss += retnet_loss.item()
        retnet_train_loss = retnet_train_loss / len(dataloader)


        with open('Retnet_training_log.txt', 'a') as log_file:
            log_file.write(f"epoch : {epoch + 1}/{num_epoch}, loss = {retnet_train_loss:.2f}\n")
        if epoch % 100 == 0:
            print("epoch : {}/{}, loss = {:.2f}".format(epoch, num_epoch, retnet_train_loss))
        if retnet_savefile and (epoch % 10 == 0):
            checkpoint = {'epoch': epoch + 1, 'embed_ae': embedAE.state_dict(), 'retnet': retnet.state_dict(), 'optimizer': retnet_optimizer.state_dict()}
            save_ckp(checkpoint, retnet_savefile)

        hap_origin = assignment(SNVdata, embedAE, retnet, num_hap=num_hap,device=device)
        hap_matrix = SNVtoHap(SNV_matrix, hap_origin.cpu().detach().numpy().astype(int), num_hap)
        mec_curr = MEC(SNV_matrix, hap_matrix)
        mec.append(mec_curr)
        if mec_curr <= mec_min:
            mec_min = mec_curr
            hap_origin_best = 1*hap_origin
            hap_matrix_best = 1*hap_matrix
            print('Epoch = %d, MEC = %d' %(epoch, mec_curr))
            deephapnet_best = {'embed_ae': embedAE.state_dict(),'retnet': retnet.state_dict()}
            torch.save(deephapnet_best, 'data/' + outhead + '/deephapnet_model')
        
    hap_origin_best = hap_origin_best.cpu().numpy()
    mec_best, hap_matrix_best,  hap_origin_best = optimise(SNV_matrix, hap_origin_best, num_hap)
    print("The MEC after refine: ",mec_best)
    if check_swer:
        np.savez('data/' + outhead + '/deephapnet', rec_hap=hap_matrix_best, rec_hap_origin=hap_origin_best, true_hap=true_haplo)
        swer_best = SWER(hap_matrix_best,true_haplo)
        return mec_best, swer_best
    else:
        np.savez('data/' + outhead + '/deephapnet', rec_hap=hap_matrix_best, rec_hap_origin=hap_origin_best)
        return mec_best

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filehead", help="Prefix of required files", type=str, required=True)
    parser.add_argument("-p", "--ploidy", help="Ploidy of organism", default=2, type=int)
    parser.add_argument("-a", "--algo_runs", help="Number of experimental runs per dataset", default=1, type=int)
    parser.add_argument("-g", "--gpu", help='GPU to run DeepHapNet', default=-1, type=int)
    parser.add_argument("--set_seed", help="True for set seed",action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = parser()
    if args.set_seed:
        setting_seed()
    fhead = args.filehead
    mec = []
    best_mec = float('inf')
    for r in range(args.algo_runs):
        print('RUN %d for %s' % (r+1, fhead))
        mec_r = train_deephapnet(fhead, num_epoch=2000, gpu=args.gpu, num_hap=args.ploidy)
        if mec_r < best_mec:
            best_mec = mec_r
            shutil.copy('data/' + fhead + '/deephapnet.npz', 'data/' + fhead + '/deephapnet_best.npz')                
        mec.append(mec_r)

    print('MEC scores for DeepHapNet: ', mec)
    print('Best MEC: %d' % mec[np.argmin(mec)])

