import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train supervised mlp
def train_mlp(model, optimizer, n_epochs, loss_fc, data_loader, device):  
    model.train()
    with tqdm(range(n_epochs), desc='training model', unit='epoch') as tepoch:
        for t in tepoch:
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)  
                # compute outputs
                outputs = model(X)    
                # compute loss and backprop
                optimizer.zero_grad()
                loss = loss_fc(outputs, y)
                loss.backward()
                optimizer.step()

# train dllp
def train_dllp(model, optimizer, n_epochs, loss_fc, data_loader, device):
    model.train()    
    with tqdm(range(n_epochs), desc='training model', unit='epoch') as tepoch:
        for i in tepoch:
            for X, y, bags, bag_prop in data_loader:
                # prepare bag data
                X, bag_prop = X[0].to(device), bag_prop[0].to(device)
                # compute outputs
                batch_avg, outputs = model(X) 
                # compute loss and backprop
                optimizer.zero_grad()
                loss = loss_fc(batch_avg, bag_prop)
                loss.backward()
                optimizer.step()    


    




    


