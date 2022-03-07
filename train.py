import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from tqdm import tqdm

import torch.nn.functional as F
import torchinfo

from model import Heartbeat, CNN_Heart
from torch.utils.data import DataLoader

# LOad Data
train = pd.read_csv('./ECG_Kaggle/mitbih_train.csv', header=None)
print("The shape of train dataset : ",train.shape)
train.head()


test = pd.read_csv('./ECG_Kaggle/mitbih_test.csv',header=None)
print("The shape of test dataset : ",test.shape)
test.head()

cols = [*range(187),'target']
train.columns = cols
test.columns = cols




#### Datasets

train_ds = Heartbeat(train)
test_ds = Heartbeat(test)
l, _ = train_ds[0]
seq_len = l.shape[0]
n_classes = 5

#### Dataloader

train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True)



def train_model_dl(model, train_dl, test_dl, n_epochs):
  # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  # criterion = nn.L1Loss(reduction='sum').to(device)
  
  history = dict(train=[], val=[])
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for i, (seq, target) in enumerate(train_dl):
      optimizer.zero_grad()
      seq = seq[:, None, :].to(device).float()
      target = target.to(device)
      # seq = seq.float()
      # target.to(device)
      output = model(seq)
      # print('output ', output.shape, 'target  ', target.shape, 'seq  ', seq.shape)
      # print('output ', output, 'target  ', target)

      loss = criterion(output, target)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for i, (seq, target) in enumerate(test_dl):
        seq = seq[:, None, :].to(device).float()
        # seq = seq.to(device)
        # seq = seq.float()
        target = target.to(device)
        output = model(seq)
        loss = criterion(output, target)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model, './model_best.pth')

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    if epoch > 2:
        if history['train'][epoch-1] >= history['train'][epoch-2] and history['train'][epoch-1] >= history['train'][epoch-3]:
            print(f"The convergence of the Neural Network has stalled, exiting the loop....")
            break
  model.load_state_dict(best_model_wts)
  return model.eval(), history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_Heart(seq_len, n_classes)
model = model.to(device)

model, history = train_model_dl(model, train_dataloader, test_dataloader, n_epochs=30)

torch.save(model, './model_cnn1.pth')

###### Check accuracy

model.to(device)
accuracy = []

with torch.no_grad():
  for i, (beats, target) in enumerate(test_dataloader):
    # print(beats.shape)
    beats = torch.unsqueeze(beats, dim =1).to(device)
    # print(beats.shape)
    y_pred = np.zeros(r, dtype = int)
    ps = model.forward(beats)
    ps = F.softmax(ps, dim = 1)
    y_pred = torch.argmax(ps,dim=1).to('cpu')
    y_pred = y_pred.numpy()
    y_test = target.numpy()    
    # print(y_test, y_pred, np.mean(y_test == y_pred))

    accuracy.append(np.mean(y_test == y_pred))
    
print('Accuracy of the model is: ', np.mean(accuracy))