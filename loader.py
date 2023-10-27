import numpy as np
from torch.utils.data import Dataset

# dataset class for supervised mlp
class MLPDataset(Dataset):
    def __init__(self, data_df=None):
        self.data_df = data_df
        self.data = self.data_df.drop(['target', 'bag'], axis=1).to_numpy()
        self.y = self.data_df['target'].to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

    def get_n_features(self):
        return len([col for col in self.data_df.columns.tolist() if col not in ['target', 'bag']])
    
    def get_n_classes(self):
        return len(np.unique(self.data_df['target'].to_numpy()))
    
# dataset class for regular llp datasets  
class LLPDataset(Dataset):
    def __init__(self, data_df=None):
            self.data_df = data_df

    def __len__(self):
        return len(np.unique(self.data_df['bag'].to_numpy()))

    def __getitem__(self, idx):
        bag_data = self.data_df[self.data_df['bag'] == idx]
        targets = bag_data['target'].to_numpy()
        bags = bag_data['bag'].to_numpy()
        data = bag_data.drop(['target', 'bag'], axis=1).to_numpy()
        bag_prop = bag_data['target'].value_counts().sort_index().to_numpy() / len(bags)
        return data, targets, bags, bag_prop
    
    def get_n_classes(self):
        return len(np.unique(self.data_df['target'].to_numpy()))

    def get_n_features(self):
        return len([col for col in self.data_df.columns.tolist() if col not in ['target', 'bag']])
