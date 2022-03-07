import torch.nn as nn
import torch.nn.functional as F 
import torch
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset

# Approach CNN 1_D

################b 2 Layers ###########################
class CNN_Heart(nn.Module):
	
	def __init__(self, seq_len, n_classes):
		super(CNN_Heart, self).__init__()

		self.fc1 = nn.Linear(seq_len, 128) 
		self.conv1 = nn.Conv1d(1, 16, 3)
    	
		self.conv2 = nn.Conv1d(16, 16, 3)

		self.pool1 = nn.MaxPool1d(2,stride = 2)
        # # in_channels = 6 because self.conv1 output 6 channel
		# # 5*5 comes from the dimension of the last convnet layer
		self.conv3 = nn.Conv1d(16, 16, 3)

		self.pool2 = nn.MaxPool1d(2,stride = 2)
		self.fc2 = nn.Linear(480, 64)
		self.fc3 = nn.Linear(64,n_classes)
        
	def forward(self, x): 
		x = F.relu(self.fc1(x))
		x = F.relu(self.conv1(x))
		x = self.pool1(F.relu(self.conv2(x)))
		x = self.pool2(F.relu(self.conv3(x)))
		x = x.view(-1, 16*30)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x


class Heartbeat(Dataset):
    def __init__(self, data_df):
        self.target = data_df['target'].to_numpy()
        self.beat = data_df.drop(labels='target', axis=1)
        
    def __len__(self):
        return len(self.beat)

    def __getitem__(self, idx):
            beat = self.beat.iloc[idx,:]
            target = self.target[idx]
            return torch.tensor(beat).type(torch.float32), torch.tensor(target).type(torch.LongTensor)