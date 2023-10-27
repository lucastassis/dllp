import torch
from torch import nn
torch.set_default_dtype(torch.float64)

# supervised mlp
class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(MLP, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(nn.LogSoftmax(dim=1))

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x

# dllp
class BatchAvgLayer(nn.Module):
    def __init__(self):
        super(BatchAvgLayer, self).__init__()

    def forward(self, x):
        return torch.mean(input=x, dim=0)

class MLPBatchAvg(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(MLPBatchAvg, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(nn.LogSoftmax(dim=1))
        self.batch_avg = BatchAvgLayer()

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        softmax = x.clone()
        if self.training:
            x = self.batch_avg(x)
        return x, softmax
