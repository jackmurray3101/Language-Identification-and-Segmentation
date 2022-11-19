import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# arg1: log file to open 
# arg2: number of epochs to be included in output (should be <= number of epochs the training ran for)

filename = sys.argv[1]
specified_epochs = int(sys.argv[2])

f = open(filename, "r")

train = np.zeros(specified_epochs)
val = np.zeros(specified_epochs)
epochs = np.linspace(0, specified_epochs-1, specified_epochs)

epoch = 0
for line in f:
  if re.search("average_loss_per_batch:", line):
    line = line.split(",")
    epoch = re.findall("[0-9]+", line[0])
    epoch = epoch[0]
    if (int(epoch) >= specified_epochs):
      break
    train_accuracy = re.findall("[0-9]{2}\.{1}[0-9]{2}", line[3])
    train_accuracy = train_accuracy[0]
    train[int(epoch)] = train_accuracy

  elif re.search("validation accuracy", line):
    line = line.split(",")
    val_accuracy = re.findall("[0-9]{2}\.{1}[0-9]{2}", line[1])
    val_accuracy = val_accuracy[0]
    val[int(epoch)] = val_accuracy

f.close()
out_filename = re.sub(".gadi-pbs.log", f"_plot_{specified_epochs}epochs.png", filename)
print(train)
print(val)
plt.plot(epochs, train, label="Training")
plt.plot(epochs, val, label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.xlim([0, specified_epochs - 1])
plt.ylim([0, 100])
plt.legend(loc="lower right")
plt.savefig(out_filename, dpi=250)
