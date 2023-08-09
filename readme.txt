What is this repository for?

This is the code for "A Lightweight 1D Deep Convolutional Neural Network for MMP Prediction in CO2-EOR Process" The codes will be included after final acceptance for publishing the related scientific paper.

How do I get set up?

The complete source code are provided in main.py. Our code needs python3 environment in Windows 64bit. Besides, the necessary toolkits of python3 are as follows: pandas 1.5.3, numpy1.24.2, matplotlib3.7.1, sklearn1.2.2, torch 1.13.1. You shoud better get these toolkits firstly.

Data quesetion

We have provided the real MMP values from experiments in Realvalue.csv. The main variables for predicting MMP values are provided in MMP.csv

Usage:
(1) Choose one of the following:
	1. Complete training and testing process: Open the files through pycharm and run main.py.
	2. Verify using only tested modules: Open the files through pycharm and run testdemo.py.
(2) Some code considerations:
	Our testdemo.py utilize the preserved weights of training process, which are provided in this file as e1.pth and e2.pth. 
	If you run the main.py, after some seconds, the training and testing results will be displayed separately in your running area. And the weights of training process will preserve in your computer as e1.pth and e2.pth.
 	