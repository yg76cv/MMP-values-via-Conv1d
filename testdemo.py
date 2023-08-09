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



class CnnRegressor(torch.nn.Module):
  def __init__(self, batch_size, inputs, outputs):
    super(CnnRegressor, self).__init__()
    self.batch_size = batch_size
    self.inputs = inputs
    self.outputs = outputs
    self.input_layer = Conv1d(inputs, batch_size,2,1,1)
    self.max_pooling_layer= MaxPool1d(1)
    #adding 2 layers
    #using 512 neurons
    self.conv_layer = Conv1d(batch_size, 96, 2,1,1)
    self.conv_layer1 = Conv1d(96, 96, 2,1,1)
    self.conv_layer2 = Conv1d(96, 96, 2,1,1)
    self.dropout=nn.Dropout(p=0.2)
    self.conv_layer3 = Conv1d(96, 96, 2,1,1)
    self.flatten_layer = Flatten()    #扁平化 变成一维
    self.linear_layer = Linear(576,12)
    self.outputs_layer = Linear(12, outputs)
  def feed(self, input):
    input = input.reshape((self.batch_size, self.inputs, 1))
    output = relu(self.input_layer(input))
    output = self.max_pooling_layer(output)
    output = relu(self.conv_layer(output))
    output = relu(self.conv_layer1(output))
    output = relu(self.conv_layer2(output))
    output = self.dropout(output)
    output = relu(self.conv_layer3(output))
    output = self.flatten_layer(output)
    output = self.linear_layer(output)
    output = self.outputs_layer(output)
    return output
batch_size = 12
model = CnnRegressor(batch_size, 12,1)
##定义评估指标：
def aee(y_true, y_pred):
  n = len(y_true)
  t = sum((y_true - y_pred) / y_true) * 100 / n
  return t

def mape(y_true, y_pred):
    r = len(y_true)
    n =sum(abs((y_pred - y_true) / y_true)) * 100/r
    return n

def smape(y_true, y_pred):
    r = len(y_true)
    m = sum(abs(y_pred - y_true) / (abs(y_pred) + abs(y_true))) * 200 / r
    return m
#定义模型
def model_loss(model, dataset, train=False, optimizer=None):
  performance = torch.nn.MSELoss()
  maerror = torch.nn.L1Loss()
  score_metric = R2Score()
  avg_loss = 0
  avg_m = 0
  avg_sm = 0
  avg_error = 0
  avg_rerror = 0
  avg_score = 0
  count = 0

  for input, output in iter(dataset):
    predictions = model.feed(input)
    loss = performance(predictions, output)
    error = maerror(predictions, output)
    rerror = aee(output, predictions)
    m = mape(output, predictions)
    q = smape(output, predictions)
    score_metric.update([output, predictions])
    score = score_metric.compute()
    if (train):
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    avg_loss += loss.item()
    avg_rerror += rerror.item()
    avg_m += m.item()
    avg_sm += q.item()
    avg_error += error.item()
    avg_score += score
    count += 1
    if count == 1:
      pre = model.feed(input).detach()
      out = output
    p = predictions.detach()
    pre = torch.cat((pre, p))
    out = torch.cat((out, output))
  return avg_loss / count, avg_error / count, avg_rerror / count, avg_score / count, avg_m / count, avg_sm / count, pre, out

import time


model_loss=torch.load('e1.pth')##读取权重文件
model=torch.load('e2.pth')
##test data(random sample)
inputs=torch.tensor([[4.5000e+01, 4.5000e+01, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.8830e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [4.5000e+01, 4.5000e+01, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.0830e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [8.0000e+01, 0.0000e+00, 2.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.0556e+01, 1.4280e+01, 7.0220e+01, 2.1580e+02, 1.0500e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 6.7778e+01, 2.2900e+01, 4.1100e+01, 2.3181e+02, 3.1000e+01],
        [8.9080e+01, 0.0000e+00, 7.8200e+00, 0.0000e+00, 1.9000e-01, 2.9100e+00,
         0.0000e+00, 6.0000e+01, 2.2780e+01, 1.1190e+01, 2.3270e+02, 1.3530e+01],
        [7.5000e+01, 2.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.8330e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [9.0840e+01, 0.0000e+00, 7.9700e+00, 0.0000e+00, 2.0000e-01, 9.9000e-01,
         0.0000e+00, 8.0000e+01, 1.9800e+00, 4.4700e+00, 2.3550e+02, 1.2150e+01],
        [7.5000e+01, 2.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 3.6250e+01, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.4444e+01, 3.8400e+01, 5.1200e+01, 2.1383e+02, 5.4000e+00],
        [5.0000e+01, 5.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.8330e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [9.1750e+01, 0.0000e+00, 8.0500e+00, 0.0000e+00, 2.0000e-01, 0.0000e+00,
         0.0000e+00, 6.0000e+01, 1.9800e+00, 4.4700e+00, 2.3550e+02, 1.2150e+01],
        [9.0000e+01, 0.0000e+00, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.9750e+01, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [9.0000e+01, 0.0000e+00, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.4444e+01, 4.0300e+01, 2.5400e+01, 2.1383e+02, 2.9300e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 6.3000e+01, 1.8620e+01, 9.6100e+00, 2.1500e+02, 9.5600e+00],
        [9.0000e+01, 0.0000e+00, 0.0000e+00, 1.0000e+01, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.8889e+01, 2.2820e+01, 3.7840e+01, 2.1527e+02, 3.4340e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.9000e+01, 1.0300e+01, 1.6290e+01, 2.0300e+02, 6.1300e+00],
        [5.0000e+01, 5.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.1450e+01, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [9.0000e+01, 0.0000e+00, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.7222e+01, 5.1100e+00, 8.9890e+01, 2.1100e+02, 0.0000e+00],
        [9.0840e+01, 0.0000e+00, 7.9700e+00, 0.0000e+00, 2.0000e-01, 9.9000e-01,
         0.0000e+00, 8.0000e+01, 2.2780e+01, 1.1190e+01, 2.3270e+02, 1.3530e+01],
        [9.1750e+01, 0.0000e+00, 8.0500e+00, 0.0000e+00, 2.0000e-01, 0.0000e+00,
         0.0000e+00, 8.0000e+01, 2.2780e+01, 1.1190e+01, 2.3270e+02, 1.3530e+01],
        [9.0840e+01, 0.0000e+00, 7.9700e+00, 0.0000e+00, 2.0000e-01, 9.9000e-01,
         0.0000e+00, 1.0000e+02, 2.3800e+01, 6.7900e+00, 2.2820e+02, 2.4680e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 7.5850e+01, 2.2880e+01, 8.4600e+00, 2.3991e+02, 8.6800e+00],
        [9.3210e+01, 0.0000e+00, 0.0000e+00, 4.0000e-01, 1.0000e-02, 3.8000e-01,
         6.0000e+00, 8.5560e+01, 9.4400e+00, 3.9600e+00, 2.6800e+02, 4.4530e+01],
        [9.0000e+01, 0.0000e+00, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.4444e+01, 3.1800e+01, 3.3700e+01, 1.9920e+02, 2.9500e+01],
        [7.5000e+01, 2.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.0556e+01, 1.4280e+01, 7.0220e+01, 2.1580e+02, 1.0500e+01],
        [9.2250e+01, 0.0000e+00, 0.0000e+00, 7.7500e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.1222e+02, 2.8100e+01, 3.4200e+01, 2.4150e+02, 3.2700e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.9000e+01, 1.0470e+01, 8.4600e+00, 2.3000e+02, 5.4500e+00],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.1778e+02, 2.1420e+01, 6.1100e+00, 2.9225e+02, 3.4940e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 8.0000e+01, 8.6000e+00, 3.3040e+01, 2.6870e+02, 5.3360e+01],
        [6.7500e+01, 2.2500e+01, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 2.4350e+01, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 4.0560e+01, 2.2250e+01, 9.4900e+00, 2.2900e+02, 9.0100e+00],
        [4.0000e+01, 4.0000e+01, 2.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.6650e+01, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [8.0000e+01, 0.0000e+00, 2.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 8.3500e+00, 1.6460e+01, 7.1600e+00, 2.4000e+02, 1.6480e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.1222e+02, 2.8100e+01, 3.4200e+01, 2.4150e+02, 3.2700e+01],
        [7.5000e+01, 2.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 1.1556e+02, 1.9180e+01, 6.9100e+00, 2.8100e+02, 6.7400e+00],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.7222e+01, 3.4800e+00, 5.9640e+01, 2.1100e+02, 3.1880e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 3.9444e+01, 2.1810e+01, 4.5350e+01, 2.2770e+02, 2.7840e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 6.1111e+01, 3.8400e+01, 5.1200e+01, 2.1383e+02, 5.4000e+00],
        [6.7500e+01, 2.3000e+01, 1.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.8830e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 9.0556e+01, 1.7970e+01, 5.5500e+00, 2.5614e+02, 4.0080e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 3.2200e+01, 1.4280e+01, 1.0740e+01, 2.0600e+02, 1.0500e+01],
        [5.0000e+01, 5.0000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 5.7222e+01, 1.4280e+01, 7.0220e+01, 2.1580e+02, 1.0500e+01],
        [8.9080e+01, 0.0000e+00, 7.8200e+00, 0.0000e+00, 1.9000e-01, 2.9100e+00,
         0.0000e+00, 8.0000e+01, 2.3800e+01, 6.7900e+00, 2.2820e+02, 2.4680e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 9.0350e+01, 2.8590e+01, 7.9500e+00, 2.6490e+02, 2.7790e+01],
        [1.0000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 7.9350e+01, 2.8590e+01, 7.9500e+00, 2.6490e+02, 2.7790e+01]])


outputs=torch.tensor([[10.3800],
        [ 8.8300],
        [14.8720],
        [16.8922],
        [ 9.1635],
        [10.3500],
        [16.1274],
        [ 7.5156],
        [ 9.4803],
        [ 8.9700],
        [16.2101],
        [11.0320],
        [13.1000],
        [14.2000],
        [ 9.9974],
        [12.0000],
        [ 6.6192],
        [16.2027],
        [11.4664],
        [12.6937],
        [17.7684],
        [ 8.6950],
        [34.4750],
        [13.1000],
        [ 7.5291],
        [19.6845],
        [11.7000],
        [31.0300],
        [26.7517],
        [10.2736],
        [ 8.2740],
        [12.1352],
        [14.8243],
        [24.1454],
        [23.8222],
        [13.7895],
        [13.7895],
        [10.3421],
        [12.4100],
        [27.6834],
        [ 6.9000],
        [ 8.9632],
        [14.6932],
        [11.2250],
        [11.2950]])

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