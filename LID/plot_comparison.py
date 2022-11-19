'''
  Given text files with output training data from multiple models, plot
  the training/validation curves on the same graph for comparison
'''

import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# arg1: log file to open 
# arg2: number of epochs to be included in output (should be <= number of epochs the training ran for)

cwd = os.path.join(os.getcwd(), "stage8")
filename1 = os.path.join(cwd, "w2vsid.txt")
filename2 = os.path.join(cwd, "w2vbase.txt")
filename3 = os.path.join(cwd, "xlsr.txt")
filename4 = os.path.join(cwd, "d2v.txt")
filename5 = os.path.join(cwd, "hbbase.txt")
filename6 = os.path.join(cwd, "hbsid.txt")
label1 = "wav2vec2-base-SID"
label2 = "wav2vec2-base"
label3 = "wav2vec2-large-XLSR"
label4 = "data2vec-base"
label5 = "HuBERT-base-ls960"
label6 = "HuBERT-base-SID"

specified_epochs = 50

f1 = open(filename1, "r")
f2 = open(filename2, "r")
f3 = open(filename3, "r")
f4 = open(filename4, "r")
f5 = open(filename5, "r")
f6 = open(filename6, "r")

train1 = np.zeros(specified_epochs)
val1 = np.zeros(specified_epochs)

train2 = np.zeros(specified_epochs)
val2 = np.zeros(specified_epochs)

train3 = np.zeros(specified_epochs)
val3 = np.zeros(specified_epochs)

train4 = np.zeros(specified_epochs)
val4 = np.zeros(specified_epochs)

train5 = np.zeros(specified_epochs)
val5 = np.zeros(specified_epochs)

train6 = np.zeros(specified_epochs)
val6 = np.zeros(specified_epochs)

epochs = np.linspace(0, specified_epochs-1, specified_epochs)

epoch = 0
for line in f1:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train1[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val1[int(epoch)] = val_accuracy

f1.close()
epoch = 0
for line in f2:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train2[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val2[int(epoch)] = val_accuracy

f2.close()
epoch = 0
for line in f3:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train3[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val3[int(epoch)] = val_accuracy

f3.close()

epoch = 0
for line in f4:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train4[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val4[int(epoch)] = val_accuracy

f4.close()

epoch = 0
for line in f5:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train5[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val5[int(epoch)] = val_accuracy

f5.close()

epoch = 0
for line in f6:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train6[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{1,3}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val6[int(epoch)] = val_accuracy

f6.close()


out_filename = "comparison.png"

plt.plot(epochs, val1, label=label1)
plt.plot(epochs, val2, label=label2)
plt.plot(epochs, val3, label=label3)
plt.plot(epochs, val4, label=label4)
plt.plot(epochs, val5, label=label5)
plt.plot(epochs, val6, label=label6)
plt.title("Comparison of validation accuracy on 14 languages")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.xlim([0, specified_epochs - 1])
plt.ylim([0, 100])
plt.legend(loc="lower right")
plt.savefig(out_filename, dpi=250)
