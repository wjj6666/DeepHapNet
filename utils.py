import numpy as np
from os import path
from scipy.sparse import coo_matrix, csr_matrix
from itertools import permutations, product
from typing import List, Optional
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch

# get the ACGT statistics of a read matrix
def ACGT_count(submatrix):
	"""
	submatrix:
		Read-SNV marix (m x n)

	Returns
		(n x 4) matrix of base counts at each SNP
	"""
	out = np.zeros((submatrix.shape[1], 4))
	for i in range(4):
		out[:, i] = (submatrix == (i + 1)).sum(axis = 0)

	return out

def SNVtoHap(SNV_matrix, origin,num_hap: int=2):    
    """
    SNV_matrix:
        Full read-SNV matrix
    origin: 
        Specifies origin of each read by an int from (0, 1, ..., num_hap-1)
        
    Returns
        matrix of haplotypes (haplotypes x SNPs)
    """
    
    origin_val = np.unique(origin)
    accepted_val = np.arange(num_hap)
    if np.any(np.intersect1d(origin_val, accepted_val) != origin_val):
    	raise ValueError("Invalid origin values passed as argument.")

    hap_matrix = np.zeros((num_hap, SNV_matrix.shape[1]), dtype=int)
    ACGTcount = ACGT_count(SNV_matrix)  # Stats of entire read matrix
    for h in range(num_hap):
        reads_h = SNV_matrix[origin == h]  # Reads attributed to haplotype i
        h_stats = np.zeros((SNV_matrix.shape[1], 4))
        
        if len(reads_h) != 0:
            h_stats = ACGT_count(reads_h) # ACGT statistics of a single nucleotide position
        hap_matrix[h, :] = np.argmax(h_stats, axis = 1) + 1  # Most commonly occuring base at each pos  
        
        uncov_pos = np.where(np.sum(h_stats, axis = 1) == 0)[0]  # Positions uncovered by reads
        for j in range(len(uncov_pos)):  # if not covered, select the most doninant one based on 'ACGTcount'  
            base_max = np.flatnonzero(ACGTcount[uncov_pos[j], :] == np.amax(ACGTcount[uncov_pos[j], :])) + 1
            if len(base_max) == 1:  # Single dominant base
                hap_matrix[h, uncov_pos[j]] == base_max[0]
            else:  # Choose one of the dominant bases at random
                hap_matrix[h, uncov_pos[j]] = np.random.choice(base_max)

    return hap_matrix
# calculate hamming distance

def HM_distance(read: np.ndarray, 
	haplo: np.ndarray) -> int:
    """
    read:
        Read denoted by base at each SNP (1-D numpy array)
	haplo:
		Haplotype denoted by base at each SNP (1-D numpy array)

	Returns
		Hamming distance between read and haplotype 

	"""
    if isinstance(read, np.matrix):
        read = read.A1
    if isinstance(haplo, np.matrix):
        haplo = haplo.A1

    if np.shape(read) != np.shape(haplo):
        # print(read.shape, haplo.shape)
        raise ValueError('Read and haplotype must be of the same dimension.')

    return sum((haplo - read)[read != 0] != 0)


def MEC(SNV_matrix: np.ndarray,
        hap_matrix: np.ndarray) -> int:  # Compute MEC score
    
    """
	SNV_matrix:
		Read-SNP matrix
	hap_matrix:
		Haplotype-SNP matrix

	Returns
		MEC score for given SNV matrix and haplotypes

    """

    if np.shape(SNV_matrix)[1] != np.shape(hap_matrix)[1]:
        raise ValueError("Different number of SNPs in reads and haplotypes.")
    
    res = 0
    # print(type(SNV_matrix), type(hap_matrix))
    for SNV_read in SNV_matrix:
        dis = [HM_distance(SNV_read.squeeze(), hap) for j, hap in enumerate(hap_matrix)]
        res = res + min(dis)
        
    return res



def read_true_hap(gt_file: str, pos_file: str) -> np.ndarray:
    try:
        with open(pos_file, "r") as pos_f:
            pos_str = pos_f.readline().split()
            pos = np.array([int(ps) - 1 for ps in pos_str]) 

        with open(gt_file, "r") as gt_f:
            lines = gt_f.readlines()
            data_lines = [line.strip() for line in lines if not line.startswith(">")]
            true_hap = np.zeros((len(data_lines), len(pos)), dtype=int)
            for i, line in enumerate(data_lines):
                for j, p in enumerate(pos):
                    base = line[p]
                    if base == 'A':
                        true_hap[i, j] = 1
                    elif base == 'C':
                        true_hap[i, j] = 2
                    elif base == 'G':
                        true_hap[i, j] = 3
                    elif base == 'T':
                        true_hap[i, j] = 4
                    elif base == '-':
                        true_hap[i, j] = 0
                    else:
                        true_hap[i, j] = -1 
        return true_hap
    except FileNotFoundError as e:
        raise OSError(f"File not found: {e.filename}")


def read_hap(hap_file: str) -> np.ndarray:
    """
    hap_file:
        File containing haplotypes
    
    Returns
        k x n numpy array of haplotypes
    
    """
    
    if not path.isfile(hap_file):
        raise OSError("Haplotype file not found.")
    
    base_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 0}  # mapping of base to int
    hap_list = []
    with open(hap_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                if 'Hap' not in line and 'hap' not in line.strip():
                    hap = np.array([base_int[c] for c in line.strip()])
                    hap_list.append(hap.astype('int'))
    
    return np.array(hap_list)		



def read_sparseSNVMatrix(mat_file: str) -> csr_matrix:
    """
    Read read-SNP matrix into sparse matrix. Useful for large matrices.

    mat_file: str
        Path to file containing read-SNP matrix
    
    Returns
        csr_matrix: Sparse read-SNP matrix
    """

    with open(mat_file, "r") as f:
        vals, rows, cols = [], [], []
        idx_val_dict = {}
        nSNV = int(f.readline().strip())
        nReads = 0
        for line in f:
            line = line.strip()
            ind_vals = line.split()
            for iv in ind_vals:
                snv, val = iv.split(",")
                vals.append(int(val))
                rows.append(nReads)
                cols.append(int(snv))
                idx_val_dict[(nReads, int(snv))] = int(val)
            nReads = nReads + 1
    
    SNV_matrix = coo_matrix((vals, (rows, cols)), shape=(nReads, nSNV)).tocsr()
    
    return SNV_matrix

class PermDict:
    """
    This is a class to represent permutations when certain indices are interchangeable due to
    the underlying object. For instance, permutations of [0, 1, 2] and [2, 1, 0] are equivalent
    on an object [A, C, A].
    
    tups: Possible values for permutation indices. Values in same list are interchangeable.
    pos: # Position in the permutation of interchangeable values
    
    For the above example, tups = [[0, 2], [1]], pos = [[0, 2], [1]]. The values [0, 2] are
    interchangeable and occur at the 0-th and 2nd indices in the permutation.
    """
    
    def __init__(self, perm_vals: List[list], pos_idx: List[List], numel: Optional[int]=None):
        self.tups = perm_vals  # Possible values for permutation indices. Values in same tuple are interchangeable
        self.pos = pos_idx  # Position in the permutation of interchangeable values
        if numel is None:
            pos_list = [p for pl in pos_idx for p in pl]
            self.n = max(pos_list) + 1
        else:
            self.n = numel
    
def gen_perm(perm: PermDict):  # Generate next possible permutation for given configuration
    
    """
    This is a generator function that generates all possible permutations given a PermDict object.
    In effect, this generates all possible permutations by interchanging positions where equal
    values exist in the object on which the permutations are invoked.
    
    Parameters
    ----------
    perm: PermDict
        PermDict object to generate permutations from
    
    Returns
    -------
        NDArray: Permutation
    """
    
    gen_list = [permutations(v) for v in perm.tups]
    gen_perm_tup = product(*gen_list)
    for perm_tup in gen_perm_tup:
        perm_res = np.zeros(perm.n, dtype=int)
        for tup, pos in zip(perm_tup, perm.pos):
            perm_res[np.array(pos)] = np.array(tup)
        yield perm_res
        
def permute_distance(perm1: PermDict, perm2: PermDict
                    ) -> int:
    """
    This function computes the distance between the specified permutations.
    Each permutation of size n is represented as a list of tuples, where each
    tuple represents positions where the bases/values are identical in the
    underlying haplotype.
    
    Parameters
    ----------
    perm1, perm2: PermDict
        PermDict objects between which to compute distance
    
    Returns
    -------
    int: Vector distance between closest permutations possible from inputs
    
    """
    if perm1.n != perm2.n:
        raise ValueError('Permutations are of different sizes.')
        
    res = np.inf
    for p1, p2 in product(gen_perm(perm1), gen_perm(perm2)):
        dis = np.sum(p1 != p2)
        if dis < res:
            res = dis
    return res

def SWER(recovered_haplo: np.ndarray,
                true_haplo: np.ndarray) -> float:
    """
    Function to compute vector error rate betweeen the recovered and true haplotypes.
    
    Parameters
    ----------
    recovered_haplo:
        k x n matrix of recovered haplotypes
    true_haplo:
        True haplotypes (ground truth)
    
    Returns
    -------
        float: Vector error rate for the polyploid case, switch error rate for diploids
    """
    
    if np.shape(recovered_haplo) != np.shape(true_haplo):
        raise ValueError("Input arguments should have the same shape.")
    
    n_hap, n_SNP = np.shape(true_haplo)
    if n_SNP <= 1:
        raise ValueError('Haplotypes must have more than one SNP to compute vector error.')
    
    PermDict_list = []
    vec_err = 0  # Number of vector errors
    mismatch_SNP = []
    for j in range(n_SNP):
        thap = true_haplo[:, j]
        rhap = recovered_haplo[:, j]
        perm_poss = permutations(range(n_hap))  # Possible permutations
        dis_min = np.inf
        for p_i in perm_poss:  # Finding best permutation to match thap & rhap
            dis = np.sum(rhap != thap[np.array(p_i)])
            if dis == 0:  # Perfect match
                perm_best = np.array(p_i)
                dis_min = 0
                break
            elif dis < dis_min:  # Improved match
                dis_min = dis
                perm_best = np.array(p_i)

        unique_vals, inverse_idx = np.unique(rhap, return_inverse=True)
        pos_idx = [list(np.nonzero(inverse_idx == i)[0])
                   for i in range(len(unique_vals))]
        perm_tups = [list(perm_best[np.nonzero(inverse_idx == i)])
                     for i in range(len(unique_vals))]       
#         print(perm_best, perm_tups, pos_idx)
        if dis_min < n_hap:  # Ignore if there is no good match
            PermDict_list.append(PermDict(perm_tups, pos_idx, n_hap))
        else:
            PermDict_list.append(PermDict(perm_tups, pos_idx, n_hap))
            mismatch_SNP.append(j)
        
        if j > 0:  # Computing vector error using best possible matches
            err = permute_distance(PermDict_list[-2], PermDict_list[-1])
            vec_err = vec_err + err
    
    res = vec_err/(n_hap*(n_SNP - 1))
    if n_hap == 2:  # Compute switching error in case of two haplotypes
        res = res/2
    
    # print("Number of mismatched SNPs: %d" %len(mismatch_SNP))
    # print(mismatch_SNP)
    return res


def save_ckp(state, checkpoint_path):  
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path  # Save path
    torch.save(state, f_path)

def load_ckp(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # return model, optimizer, epoch value
    return model, optimizer, checkpoint['epoch']


class SNVMatrixDataset(Dataset):
    def __init__(self, SNV_file, transform=None):
        """
        SNV_file: txt file containing SNV matrix
        """
        SNV_matrix_raw = np.loadtxt(SNV_file, dtype=int)
        self.SNV_matrix = SNV_matrix_raw[np.sum(SNV_matrix_raw != 0, axis=1) > 1]
	 
    def __len__(self):
        return np.shape(self.SNV_matrix)[0]
    
    def __getitem__(self, idx):
        SNV_row = torch.from_numpy(self.SNV_matrix[idx])
        SNV_row_onehot = F.one_hot(SNV_row, 5)[:,1:]
        SNV_row_onehot = SNV_row_onehot.type(torch.float32)
        SNV_row_onehot = SNV_row_onehot.transpose(1,0)
        return SNV_row_onehot[None,:], idx  # Shape is batch x 4 x numSNP

# Dataset class to use if SNV matrix is stored in sparse format
class SparseSNVMatrixDataset(Dataset):
    def __init__(self, SNV_matrix, transform=None):
        """
        SNV_file: Sparse/dense read-SNP matrix
        """
        self.SNV_matrix = csr_matrix(SNV_matrix)
	 
    def __len__(self):
        return self.SNV_matrix.get_shape()[0]
    
    def __getitem__(self, idx):
        SNV_row = torch.from_numpy(self.SNV_matrix.getrow(idx).todense())[0]
        SNV_row_onehot = F.one_hot(SNV_row, 5)[:,1:]
        SNV_row_onehot = SNV_row_onehot.type(torch.float32)
        SNV_row_onehot = SNV_row_onehot.transpose(1,0)
        return SNV_row_onehot[None,:], idx  # Shape is batch x 4 x numSNP


def chunk_data(datapath, chunk_size=1000, overlap_frac=0.5):
    """
    Chunk read-SNP matrix into blocks of size chunk_size shifted by 
    overlap_frac*chunk_size SNPs. 
      
    """
    with open(datapath, "r") as f:
        vals, rows, cols = [], [], []
        idx_val_dict = {}
        nSNV = int(f.readline().strip())
        nReads = 0
        for line in f:
            line = line.strip()
            ind_vals = line.split()
            for iv in ind_vals:
                snv, val = iv.split(",")
                vals.append(int(val))
                rows.append(nReads)
                cols.append(int(snv))
                idx_val_dict[(nReads, int(snv))] = int(val)
            nReads = nReads + 1
    
    SNV_matrix = coo_matrix((vals, (rows, cols)), shape=(nReads, nSNV)).tocsc()   
    chunk_overlap = int(chunk_size*overlap_frac)  
    
    print("Chunking matrix")
    chunked_data = []
    for i in range(0, nSNV, chunk_size - chunk_overlap):   
        chunk = SNV_matrix[:,i:i+chunk_size].tocsr()
		# print(i, i+chunk_size, chunk.shape)
        chunk_rows = chunk.sum(axis=1).nonzero()[0]
		# print(chunk[chunk_rows, :].get_shape())
        chunked_data.append((i, SparseSNVMatrixDataset(chunk[chunk_rows, :])))   
    print("Finished chunking matrix into %d chunks" %len(chunked_data))

    return chunked_data

