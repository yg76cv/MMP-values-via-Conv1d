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
file = r'.\mmp1.csv'
dataset = pd.read_csv(file)
dataset.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9','x10','x12','x11','y']
dataset = dataset.dropna()
del dataset['y']
dataset = pd.DataFrame(dataset)
biso=pd.read_csv(r'.\mmpk.csv')
biso.columns = ['y']
y = biso['y']
x = dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=2003, shuffle = True)
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()

x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()
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
model = CnnRegressor(batch_size, x.shape[1],1)
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
#开始训练
import time
epochs = 800
print('Trainable parameters are:' + str(model.parameters()))
optimizer = Adamax(model.parameters(), lr = .001)
inputs = torch.from_numpy(x_train_np).float()
outputs = torch.from_numpy(y_train_np.reshape(y_train_np.shape[0],1)).float()
tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle= True, drop_last=True)

lossplot = []
training_startingTime = time.time()
for epoch in range(epochs):
    avg_loss, avg_mae,avg_rerror ,avg_r2_score,avg_mape,avg_smape,pr,out = model_loss(model, loader, train=True, optimizer=optimizer)
  #r = [avg_r2_score,epoch]
    lossplot.append(avg_loss)
    avg_rerror = format(avg_rerror,".2f")
    print("Epoch" + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss)+ ":\n\tMAE = " + str(avg_mae)+ ":\n\tARE = " + str(avg_rerror)+"%"+"\n\tR^2 Score = " + str(avg_r2_score))
training_endTime = time.time() - training_startingTime
print(training_endTime)
##
# x=np.arange(epochs)
# y=np.array(lossplot)
# plt.style.use('seaborn-white')
# plt.rc('font',family='Times New Roman', size=10,weight=5)
# fig=plt.figure(figsize=(4, 4), dpi=100)
# ax1 = axisartist.Subplot(fig, 111)
# fig.add_axes(ax1)
# #通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# #"-|>"代表实心箭头："->"代表空心箭头
# ax1.axis["bottom"].set_axisline_style("-|>", size = 1)
# ax1.axis["left"].set_axisline_style("-|>", size = 1)
# ax1.axis["top"].set_visible(False)
# ax1.axis["right"].set_visible(False)
# plt.plot(x,y,linewidth=1.5,color='b')
#
#
#
# plt.xlabel("Iteration step")
# plt.grid(True,linestyle="--",color='black', alpha=0.5)
# plt.ylabel("Loss")
# plt.savefig("loss.pdf",format="pdf")
# plt.show()

#
# plt.rc('font',family='Times New Roman', size=10,weight=5)
#
# fig=plt.figure(figsize=(4, 4), dpi=100)
# ax1 = axisartist.Subplot(fig, 111)
# fig.add_axes(ax1)
# #通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# #"-|>"代表实心箭头："->"代表空心箭头
# ax1.axis["bottom"].set_axisline_style("-|>", size = 1)
# ax1.axis["left"].set_axisline_style("-|>", size = 1)
# ax1.axis["top"].set_visible(False)
# ax1.axis["right"].set_visible(False)
#
# #plt.plot(range(len(out)), out,  'go',label = 'data', alpha = 0.3)  ##go 散点
# plt.scatter(pr, out, color='b',label = 'predicted',alpha = 1)
#
# plt.plot([pr.min(), pr.max()], [out.min(), out.max()], 'r--', lw=3, label = 'fitted line')
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# lims = [0,40]
# plt.xlim(lims)
# plt.ylim(lims)
#
# font={'family':'Times New Roman','weight':'normal'
#         ,'size':9}
# plt.legend(loc = 'upper left',prop=font,frameon=True)
#
# plt.grid(True,linestyle="--",color='black', alpha=0.5)
# plt.tick_params(top = 'off', right = 'off')
# plt.savefig("1trainfitten.pdf",format="pdf")


##torch.save(model_loss,'C:\\Users\\Yao\\Desktop\\r\\e1.pth') ##保存权重文件
##model_loss=torch.load('C:\\Users\\Yao\\Desktop\\r\\e1.pth')##读取权重文件

##test
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

##
# plt.rc('font',family='Times New Roman', size=10,weight=5)
#
# fig=plt.figure(figsize=(4, 4), dpi=100)
# ax1 = axisartist.Subplot(fig, 111)
# fig.add_axes(ax1)
# #通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# #"-|>"代表实心箭头："->"代表空心箭头
# ax1.axis["bottom"].set_axisline_style("-|>", size = 1)
# ax1.axis["left"].set_axisline_style("-|>", size = 1)
# ax1.axis["top"].set_visible(False)
# ax1.axis["right"].set_visible(False)
#
# #plt.plot(range(len(out)), out,  'go',label = 'data', alpha = 0.3)  ##go 散点
# plt.scatter(pr, out, color='b',label = 'predicted',alpha = 1)
#
# plt.plot([pr.min(), pr.max()], [out.min(), out.max()], 'r--', lw=3, label = 'fitted line')
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# lims = [0,40]
# plt.xlim(lims)
# plt.ylim(lims)
#
# font={'family':'Times New Roman','weight':'normal'
#         ,'size':9}
# plt.legend(loc = 'upper left',prop=font,frameon=True)
#
# plt.grid(True,linestyle="--",color='black', alpha=0.5)
# plt.tick_params(top = 'off', right = 'off')
# plt.savefig("2testfitten.pdf",format="pdf")

##绘制差异折线图和柱状图
# num_list = out-pr
# plt.rc('font',family='Times New Roman', size=10,weight=5)
# fig=plt.figure(figsize=(8,4), dpi=100)
#
# plt.style.use('seaborn-white')
# ax1 = axisartist.Subplot(fig, 111)
# fig.add_axes(ax1)
# #通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# #"-|>"代表实心箭头："->"代表空心箭头
# ax1.axis["bottom"].set_axisline_style("-|>", size = 1)
# ax1.axis["left"].set_axisline_style("-|>", size = 1)
# ax1.axis["top"].set_visible(False)
# ax1.axis["right"].set_visible(False)
#
# plt.bar(range(len(num_list)), num_list,color='b')
# lims = [-15,15]
#
# plt.ylim(lims)
# plt.grid(True,linestyle="--",color='black', alpha=0.5)
# plt.xlabel("The number of test set")
# plt.ylabel("Comparison of predicted and actual values ")
# plt.savefig("testbar.pdf",format="pdf")
# plt.show()
# fig = plt.figure(figsize=(8, 4), dpi=100)  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
# plt.style.use('seaborn-white')
# ax1 = axisartist.Subplot(fig, 111)
# fig.add_axes(ax1)
# # 通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# # "-|>"代表实心箭头："->"代表空心箭头
# ax1.axis["bottom"].set_axisline_style("-|>", size=1)
# ax1.axis["left"].set_axisline_style("-|>", size=1)
# ax1.axis["top"].set_visible(False)
# ax1.axis["right"].set_visible(False)
# line1, = ax1.plot(range(len(pr)), pr, 'b--', label='L1D$^2$CNN', linewidth=2)
#
# line3, = ax1.plot(range(len(out)), out, 'g', label='Real')
#
# # axes.grid()
# # fig.tight_layout()
# font = {'family': 'Times New Roman', 'weight': 'normal'
#   , 'size': 10}
#
# plt.legend(loc='upper left', handles=[line1, line3], prop=font, frameon=True)
# plt.xlabel("The number of test set")
# plt.ylabel("The values of MMP ")
# plt.grid(True, linestyle="--", color='black', alpha=0.5)
# plt.savefig("testzhexian.pdf", format="pdf")
# plt.show()
