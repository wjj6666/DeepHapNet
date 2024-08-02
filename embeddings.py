import os
import numpy as np
from tqdm import tqdm  

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class ReadAE(nn.Module):
    def __init__(self, nSNP: int, latent_dim: int=None):
        super().__init__()
        self.nSNP = nSNP     
        self.sparsity_target = 0.05 
        self.sparsity_weight = 0.5


        if latent_dim is None:
        	latent_dim = int(np.ceil(nSNP/4))  

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4,5), (4,1), (0, 2)),   
            nn.PReLU(),
            nn.Conv2d(32, 64, (1,5), (1,1), 'same'),
            nn.PReLU(),
            nn.Conv2d(64, 128, (1,3), (1,1), 'same'),
            nn.PReLU(),
            nn.Flatten(),   
            )

        self.fc1 = nn.Linear(128*nSNP, latent_dim)   
        self.fc2 = nn.Linear(latent_dim, 128*nSNP)
        self.act1 = nn.PReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (1,3), (1,1), (0, 1)),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, (1,5), (1,1), (0, 2)),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 1, (4,5), (4,1), (0,2)),
            # nn.PReLU()
            )

    def forward(self, x):
        x_code = self.encoder(x)    
        x_fc1 = self.fc1(x_code)
        x_flatten = self.act1(self.fc2(x_fc1))
        x_reshape = x_flatten.view(-1, 128, 1, self.nSNP)
        return x_fc1, self.decoder(x_reshape)   


def AE_train(dataset: Dataset, num_epoch: int, embed_dim: int = None,savefile: str = None) -> ReadAE:
    batch_size = int(np.ceil(len(dataset)/20))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU....')
    else:
        device = torch.device('cpu')
        print('The code uses CPU....')

    nSNP = dataset[0][0].shape[-1]
    model = ReadAE(nSNP, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for epoch in tqdm(range(num_epoch)):
        loss = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device) 
            optimizer.zero_grad()  
            x_code, recon = model(batch_data) 
            reconstruction_loss = loss_func(recon, batch_data)  
            #sparsity_penalty = model.calculate_sparsity_penalty(x_code)
            train_loss = reconstruction_loss #+sparsity_penalty
            train_loss.backward()  
            optimizer.step()      
            loss += train_loss.item()  
        loss = loss / len(data_loader)  

        with open('AE_training_log.txt', 'a') as log_file:
            log_file.write(f"Epoch: {epoch + 1}/{num_epoch}, Loss: {loss}\n")

    return model  

