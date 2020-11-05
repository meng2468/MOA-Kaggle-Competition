import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

params = {}
#Select data
params['feature_csv'] = '../processing/feature_eng_gauss_x.csv'
params['target_csv'] = '../processing/feature_eng_temp_y.csv'

#Select hyperparameters
params['dropout'] = 0.2093013646952079
params['learning_rate'] = 0.04
params['batch_size'] = 900
params['label_smoothing'] = 0.001

df_x = pd.read_csv(params['feature_csv'])
df_y = pd.read_csv(params['target_csv'])

x = torch.tensor(df_x.values(), dtype=torch.float)
y = torch.tensor(df_y.values(), dtype=torch.float)

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x