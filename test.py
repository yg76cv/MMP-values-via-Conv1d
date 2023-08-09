import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.model_selection import train_test_split
import mpl_toolkits.axisartist as axisartist
import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adamax
import torch
from ignite.contrib.metrics.regression.r2_score import R2Score


##torch.save(model_loss,'C:\\Users\\Yao\\Desktop\\r\\e1.pth') ##保存权重文件
##model_loss=torch.load('C:\\Users\\Yao\\Desktop\\r\\e1.pth')##读取权重文件



from train import x_test_np
from  train import  y_test_np
from  train import  batch_size
import time
from train import model_loss
from  train import  model

inputs = torch.from_numpy(x_test_np).float()
outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).float()

tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle=True, drop_last=True)
testing_startingTime = time.time()
epochs2 = 20
losssum = 0
r2sum = 0
rerrorsum = 0
aesum = 0
mapesum = 0
smapesum = 0
r2plot = []
for epoch in range(epochs2):
  avg_loss, avg_mae, avg_rerror, avg_r2_score, avg_mape, avg_smape, pr, out = model_loss(model, loader)
  r2plot.append(avg_loss)

  losssum += avg_loss
  aesum += avg_mae
  rerrorsum += avg_rerror
  mapesum += avg_mape
  smapesum += avg_smape
  r2sum += avg_r2_score
  avg_rerror = format(avg_rerror, ".2f")
  print("times" + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss) + ":\n\tMAE = " + str(avg_mae) + ":\n\tARE = " + str(
    avg_rerror) + "%" + "\n\tR^2 Score = " + str(avg_r2_score) + "\n\tMAPE Score = " + str(
    avg_mape) + "%" + "\n\tSMAPE = " + str(avg_smape) + "%")

rerrorsum = format(rerrorsum / 20, ".2f")
mapesum = format(mapesum / 20, ".2f")
smapesum = format(smapesum / 20, ".2f")
print("avg_loss" + ":\n\tLoss = " + str(losssum / 20))
print("avg_R^2 Score" + ":\n\tR^2 Score = " + str(r2sum / 20))
print("avg_ARE" + ":\n\tARE = " + str(rerrorsum) + "%")
print("avg_MAE" + ":\n\tMAE = " + str(aesum / 20))
print("avg_MAPE" + ":\n\tMAPE = " + str(mapesum) + "%")
print("avg_SMAPE" + ":\n\tSMAPE = " + str(smapesum) + "%")