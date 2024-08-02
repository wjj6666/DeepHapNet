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

"""
if __name__ == "__main__":
    # 创建示例的读取和单倍型数组
    read = np.array([1, 2, 3, 4, 0])
    haplo = np.array([1, 3, 4, 4, 0])

    # 调用函数计算汉明距离
    distance = HM_distance(read, haplo)

    # 打印结果
    print("汉明距离:", distance)
"""


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

"""
if __name__ == "__main__":
    # 创建示例的读取-SNP矩阵和单倍型-SNP矩阵
    SNV_matrix = np.array([
        [1, 2, 3, 4, 0],
        [2, 3, 4, 1, 0],
        [1, 1, 2, 3, 0],
        [4, 3, 2, 1, 0],
        [1, 2, 2, 3, 0]
    ])
    hap_matrix = np.array([
        [1, 2, 3, 4, 0],
        [2, 1, 3, 4, 0],
        [1, 3, 3, 4, 0],
        [4, 2, 2, 3, 0]
    ])

    # 调用函数计算MEC分数
    mec_score = MEC(SNV_matrix, hap_matrix)

    # 打印结果
    print("MEC分数:", mec_score)
"""


# evaluate the correct phasing rate
def CPR(recovered_haplo: np.ndarray, true_haplo: np.ndarray) -> float:
	# """
	# recovered_haplo:
	# 	k x n matrix of recovered haplotypes
	# true_haplo:
	# 	True haplotypes (ground truth)

	# Returns
	# 	correct phasing rate
	# """

    if np.shape(recovered_haplo) != np.shape(true_haplo):
        raise ValueError("Input arguments should have the same shape.")
    
    distance_table = np.zeros((len(recovered_haplo), len(true_haplo)))
    for i, rec_hap in enumerate(recovered_haplo):
        for j, true_hap in enumerate(true_haplo):
            distance_table[i, j] = HM_distance(rec_hap, true_hap)

    # print("Distance table")
    # print(distance_table)

    index = permutations(range(true_haplo.shape[0]))
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
	# index = (list(index))[np.argmin(np.array(distance))]  # Best one-to-one mapping
	# print(best_matching)
    cpr = 1 - min(distance) / np.size(true_haplo)

    return cpr




def read_true_hap(gt_file: str, pos_file: str) -> np.ndarray:
    try:
        # 读取 SNP 位置文件，获取 SNP 位置信息
        with open(pos_file, "r") as pos_f:
            pos_str = pos_f.readline().split()
            pos = np.array([int(ps) - 1 for ps in pos_str])  # 将位置转换为从0开始索引

        # 读取真实基因组文件，获取 SNP 数据
        with open(gt_file, "r") as gt_f:
            # 逐行读取文件内容
            lines = gt_f.readlines()
            # 过滤掉标题行
            data_lines = [line.strip() for line in lines if not line.startswith(">")]
            # 初始化存储 SNP 数据的数组
            true_hap = np.zeros((len(data_lines), len(pos)), dtype=int)
            # 遍历每个基因组
            for i, line in enumerate(data_lines):
                # 遍历 SNP 位置，提取对应的碱基信息
                for j, p in enumerate(pos):
                    base = line[p]
                    # 将碱基信息转换为对应的整数
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
                        true_hap[i, j] = -1  # 未知情况，可以根据需求修改
        return true_hap
    except FileNotFoundError as e:
        raise OSError(f"File not found: {e.filename}")

"""
# 测试函数
gt_file = "/home/wangjiaojiao/XHap-master/A/gt_file.txt"
pos_file = "/home/wangjiaojiao/XHap-master/A/SNV_pos.txt"
haplotypes = read_true_hap(gt_file, pos_file)
print(haplotypes)
"""



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



# 保存模型的状态和优化器的状态到一个文件。
def save_ckp(state, checkpoint_path):  
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path  # Save path
    torch.save(state, f_path)

# 从文件加载模型的状态和优化器的状态。
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



# 从文本文件加载SNV矩阵，并提供对这些数据的访问
class SNVMatrixDataset(Dataset):
    def __init__(self, SNV_file, transform=None):
        """
        SNV_file: txt file containing SNV matrix
        """
        SNV_matrix_raw = np.loadtxt(SNV_file, dtype=int)
        # 对原始SNV矩阵进行过滤，仅保留那些非零元素个数大于1的行。这通常用于移除缺失数据较多的记录。
        self.SNV_matrix = SNV_matrix_raw[np.sum(SNV_matrix_raw != 0, axis=1) > 1]
	 
    def __len__(self):
        return np.shape(self.SNV_matrix)[0]
    
    def __getitem__(self, idx):
        SNV_row = torch.from_numpy(self.SNV_matrix[idx])
        # 将SNV行转换为one-hot编码。由于SNV数据以整数形式编码，这里假设存在5种可能的值（包括0作为缺失数据）。[:,1:]的操作去掉了第一列，即去掉了表示缺失数据的one-hot编码列，只保留有效的碱基编码。
        SNV_row_onehot = F.one_hot(SNV_row, 5)[:,1:]
        # 将one-hot编码的数据类型转换为float32，这通常是深度学习模型期望的数据类型。
        SNV_row_onehot = SNV_row_onehot.type(torch.float32)
        # 将one-hot编码的张量进行转置，使得其形状符合模型的输入要求。
        SNV_row_onehot = SNV_row_onehot.transpose(1,0)
        # 返回处理后的SNV行（已经转换为one-hot编码并转置）和当前行的索引。这里SNV_row_onehot[None,:]增加了一个新的批次维度，以便与PyTorch的其他数据处理工具兼容。
        return SNV_row_onehot[None,:], idx  # Shape is batch x 4 x numSNP



# Dataset class to use if SNV matrix is stored in sparse format
# 以稀疏矩阵格式处理SNV数据，提供对这些数据的访问
class SparseSNVMatrixDataset(Dataset):
    def __init__(self, SNV_matrix, transform=None):
        """
        SNV_file: Sparse/dense read-SNP matrix
        """
        
        self.SNV_matrix = csr_matrix(SNV_matrix)
	 

    def __len__(self):
        return self.SNV_matrix.get_shape()[0]
    
    def __getitem__(self, idx):
        # 从稀疏SNV矩阵中获取索引为idx的行（getrow(idx)），然后将这一行转换为密集格式（todense()），最后转换为PyTorch张量。这里的[0]是因为todense()方法返回的是一个矩阵，而[0]将其转换为一维数组。
        SNV_row = torch.from_numpy(self.SNV_matrix.getrow(idx).todense())[0]
        # 将SNV_row转换为one-hot编码。这里假设存在5种可能的值（包括表示缺失数据的0）。[:,1:]操作去掉了表示缺失数据的第一列，只保留了有效的碱基编码。
        SNV_row_onehot = F.one_hot(SNV_row, 5)[:,1:]

        # 将one-hot编码的数据类型转换为float32，这是深度学习模型期望的数据类型。
        SNV_row_onehot = SNV_row_onehot.type(torch.float32)

        # 转置one-hot编码的张量，使得其形状符合模型的输入要求。
        SNV_row_onehot = SNV_row_onehot.transpose(1,0)

        # 返回处理后的SNV行（已转换为one-hot编码并转置）和当前行的索引。这里SNV_row_onehot[None,:]增加了一个新的批次维度，以便与PyTorch的其他数据处理工具兼容。
        return SNV_row_onehot[None,:], idx  # Shape is batch x 4 x numSNP

# 将SNV矩阵分割成多个块，每个块可以有一定的重叠部分
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
    
    SNV_matrix = coo_matrix((vals, (rows, cols)), shape=(nReads, nSNV)).tocsc()   # 使用coo_matrix函数（稀疏矩阵的一种格式）将这些值转换成一个压缩列稀疏矩阵（CSC格式），其中nReads和nSNV分别表示矩阵的行数和列数。
    chunk_overlap = int(chunk_size*overlap_frac)  # Number of SNPs to overlap between blocks,计算块之间的重叠SNPs数（chunk_overlap）
    
    print("Chunking matrix")
    chunked_data = []
    for i in range(0, nSNV, chunk_size - chunk_overlap):   # 以chunk_size - chunk_overlap为步长遍历SNP索引
        chunk = SNV_matrix[:,i:i+chunk_size].tocsr()
		# print(i, i+chunk_size, chunk.shape)
        chunk_rows = chunk.sum(axis=1).nonzero()[0]
		# print(chunk[chunk_rows, :].get_shape())
        chunked_data.append((i, SparseSNVMatrixDataset(chunk[chunk_rows, :])))   # 将每个块及其起始SNP索引作为元组添加到chunked_data列表中
    print("Finished chunking matrix into %d chunks" %len(chunked_data))

    return chunked_data

