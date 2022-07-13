import os
import numpy as np
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\FeatureFiles')
featureFiles = os.listdir()
print('Current Directory :', os.getcwd(), '\nDirectory Contents:\n', featureFiles)

featureMatrix = np.load('featureMatrix.npy', allow_pickle = True)
outputLabelMatrix = np.load('outputLabels.npy', allow_pickle = True)
padInfoMatrix = np.load('padInformationMatrix.npy', allow_pickle = True)

print(outputLabelMatrix.shape, featureMatrix.shape, padInfoMatrix.shape)

nLanguages = 2
maxFiles = 1000


combinedFeatureList = []
combinedOutputLabelList = []

for lanIndex in range(nLanguages):
  for fileIndex in range(maxFiles):
    combinedFeatureList.append(featureMatrix[lanIndex][fileIndex].flatten())
    combinedOutputLabelList.append(outputLabelMatrix[lanIndex][fileIndex])


combinedFeatureList = np.array(combinedFeatureList)
combinedOutputLabelList = np.array(combinedOutputLabelList)

print(combinedFeatureList.shape, combinedOutputLabelList.shape)

encodedOutputLabelList = np.zeros((combinedOutputLabelList.shape[0], combinedOutputLabelList.max() + 1))
encodedOutputLabelList[np.arange(combinedOutputLabelList.shape[0]), combinedOutputLabelList] = 1
print(encodedOutputLabelList.shape)

x_train, x_test, y_train, y_test = train_test_split(combinedFeatureList, encodedOutputLabelList, test_size=0.2, random_state = 33)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(200, input_shape = (combinedFeatureList.shape[1], ), activation = 'relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(encodedOutputLabelList.shape[1], activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs = 22, batch_size = 2000, validation_data=(x_test, y_test))
trueOutput = [np.where(r==1)[0][0] for r in encodedOutputLabelList]
#predictedOutput = model.predict_classes(combinedFeatureList)
predict_x=model.predict(combinedFeatureList) 
predictedOutput=np.argmax(predict_x,axis=1)


# Common code

cm = confusion_matrix(trueOutput, predictedOutput)

plt.figure(figsize = (15, 15))
plt.imshow(cm, cmap = plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title('Confusion matrix')
for i in range(2):
  for j in range(2):
    plt.annotate(cm[i][j], (i, j))
plt.colorbar()
plt.show()