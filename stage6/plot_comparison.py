import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# arg1: log file to open 
# arg2: number of epochs to be included in output (should be <= number of epochs the training ran for)

filename1 = sys.argv[1]
label1 = sys.argv[2]

filename2 = sys.argv[3]
label2 = sys.argv[4]

filename3 = sys.argv[5]
label3 = sys.argv[6]

specified_epochs = int(sys.argv[7])

f1 = open(filename1, "r")
f2 = open(filename2, "r")
f3 = open(filename3, "r")

train1 = np.zeros(specified_epochs)
val1 = np.zeros(specified_epochs)

train2 = np.zeros(specified_epochs)
val2 = np.zeros(specified_epochs)

train3 = np.zeros(specified_epochs)
val3 = np.zeros(specified_epochs)


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



out_filename = "comparison.png"

plt.plot(epochs, val1, label=label1)
plt.plot(epochs, val2, label=label2)
plt.plot(epochs, val3, label=label3)
plt.title("Comparison of validation accuracy on 14 languages")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.xlim([0, specified_epochs - 1])
plt.ylim([0, 100])
plt.legend(loc="lower right")
plt.savefig(out_filename, dpi=250)
