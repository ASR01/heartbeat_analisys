import copy
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

from model import CNN_Heart

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

colors = [ '#FFDD00', '#8F00FF',  '#01BEFE', '#FF006D', '#ADFF02','#FF7D00']

sns.set_palette(sns.color_palette(colors))

random_seed = 23
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.25
  )
  ax.set_title(class_name)

### DAta Load

train = pd.read_csv('./ECG_Kaggle/mitbih_train.csv', header=None)
print("The shape of train dataset : ",train.shape)
train.head()


test = pd.read_csv('./ECG_Kaggle/mitbih_test.csv',header=None)
print("The shape of test dataset : ",test.shape)
test.head()


cols = [*range(187),'target']
train.columns = cols
test.column = cols
train.head()
print(train['target'].value_counts())

target = train.iloc[:,187]
labels = ['Normal beat','Supraventricular premature beat','Premature ventricular contraction','Fusion of ventricular and normal beat','Unclassifiable beat']
 
target_class = dict(zip([0,1,2,3,4], labels))

ax = sns.countplot(target)
plt.show()

# We try to see the 

classes = target.unique().astype(int)

ncols = 2
fig, axs = plt.subplots(
  nrows=len(classes) // ncols + 1,
  ncols=ncols,
  sharey=True,
  figsize=(20, 8)
  )

# check the mean of all signals

for i, target_c in enumerate(classes):
  
  ax = axs.flat[i]
  data = train[train.target == target_c].drop(labels='target', axis=1).mean(axis=0).to_numpy()
  plot_time_series_class(data, target_class.get(target_c), ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout();
plt.show()




# Check if the values are normalized
train.describe().iloc[7,:187].max()

train_df, test_df = train_test_split( train, test_size=0.20, random_state=random_seed)
train_df.head()