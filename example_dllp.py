import numpy as np
import pandas as pd
import torch
from net import MLPBatchAvg
from loader import LLPDataset
from eval import eval_dllp
from train import train_dllp
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

# create blobs datasets
X, y = make_blobs(n_samples=1000, n_features=2, centers=3)

# generate bags randomly
bags = np.random.randint(0, 5, size=len(X))

# create dataframe (LLPDataset expects a dataframe as input, but this can be modified easily)
data_df = pd.DataFrame(X, columns=['f0', 'f1'])
data_df['target'] = y
data_df['bag'] = bags

# define train/test datasets and dataloader
train_df, test_df = train_test_split(data_df, test_size=0.2)
train_dataset = LLPDataset(data_df=train_df)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # batch_size=1 because DLLP paper suggests bag = batch
test_dataset = LLPDataset(data_df=test_df)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# define DLLP model
model = MLPBatchAvg(in_features=2, out_features=3, hidden_layer_sizes=(100,))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fc = torch.nn.KLDivLoss(reduction='batchmean')
model.to(device)

# train model
train_dllp(model=model, optimizer=optimizer, n_epochs=100, loss_fc=loss_fc, data_loader=train_dataloader, device=device)

# eval model
metrics = eval_dllp(model=model, data_loader=test_dataloader, device=device)
print(metrics)
