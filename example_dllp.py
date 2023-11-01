import numpy as np
import pandas as pd
import torch
from net import MLPBatchAvg
from loader import LLPDataset
from train import train_dllp
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

# create blobs datasets
X, y = make_blobs(n_samples=1000, n_features=2, centers=3)

# generate bags randomly
bags = np.random.randint(0, 5, size=len(X))

# split data
X_train, X_test, y_train, y_test, bags_train, bags_test = train_test_split(X, y, bags, test_size=0.2)

# create bag proportions train
proportions = []
for b in np.unique(bags_train):
    idx = np.where(bags_train == b)[0]
    bag_proportions = np.unique(y_train[idx], return_counts=True)[1] / len(idx)
    proportions.append(bag_proportions)
proportions = np.vstack(proportions)

# create train dataset
train_dataset = LLPDataset(X=X_train, bags=bags_train, proportions=proportions)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # batch_size=1 because DLLP paper suggests bag = batch

# define DLLP model
model = MLPBatchAvg(in_features=2, out_features=3, hidden_layer_sizes=(100,))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fc = torch.nn.KLDivLoss(reduction='batchmean')
model.to(device)

# train model
train_dllp(model=model, optimizer=optimizer, n_epochs=100, loss_fc=loss_fc, data_loader=train_dataloader, device=device)

# eval model
model.eval()
with torch.no_grad():
    X_test = torch.tensor(X_test).to(device)
    _, outputs = model(X_test)
    y_pred = outputs.argmax(dim=1).cpu().tolist()
acc = accuracy_score(y_test, y_pred)
print(f'accuracy: {acc * 100}%')