import os
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt
from librosa.display import specshow

samplingRate = 22050
os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\data')
languages = sorted(os.listdir())
print('Current Directory :', os.getcwd(), '\nDirectory Contents:', languages)
nLanguages = len(languages)


maxFiles = 1000 # should not be bigger than the min number of files in one of the dirs

""" for lanIndex in range(nLanguages):
  os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\data')
  os.chdir(languages[lanIndex])

  audioFilenames = os.listdir()
  print('Current Directory :', os.getcwd(), '\nNumber of Files:', len(audioFilenames))

  startTime = time.time()
  audioFeatures = {}
  
  for fileIndex in range(maxFiles):
    audioFile = audioFilenames[fileIndex]
    audioSignal = librosa.load(audioFile)[0]
    mfcc = librosa.feature.mfcc(audioSignal, samplingRate, n_mfcc = 13)
    delta = librosa.feature.delta(mfcc, order = 1)
    deltaDelta = librosa.feature.delta(mfcc, order = 2)
    audioFeatures[fileIndex] = np.concatenate((mfcc, delta, deltaDelta))

  os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\FeatureFiles')
  np.save('{0}AudioFeatures.npy'.format(languages[lanIndex]), audioFeatures, allow_pickle = True)
  totalTime = time.time() - startTime
  print('Average Time Taken per File = ', totalTime/maxFiles, 's\nTotal Time Taken = ', totalTime, 's') """


os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\FeatureFiles')
maxDim = np.NINF

for lanIndex in range(nLanguages):
  audioFeatures = np.load('{0}AudioFeatures.npy'.format(languages[lanIndex]), allow_pickle = True)[()]
  for fileIndex in range(len(audioFeatures)):
    nFrames = audioFeatures[fileIndex].shape[1]
    if maxDim < audioFeatures[fileIndex].shape[1]:
      maxDim = nFrames 

padInfoMatrix = []
outputLabelMatrix = []
featureMatrix = []

for lanIndex in range(nLanguages):
  lanPadInfoList = []
  audioFeatures = np.load('{0}AudioFeatures.npy'.format(languages[lanIndex]), allow_pickle = True)[()]
  outputLabelList = []
  featureList = []

  for fileIndex in range(len(audioFeatures)):
    audioFile = audioFeatures[fileIndex]
    lanPadInfoList.append(maxDim - audioFile.shape[1])
    audioFeatures[fileIndex] = np.pad(audioFile, ((0, 0), (0, maxDim - audioFile.shape[1])), 'constant', constant_values = 0)
    featureList.append(audioFeatures[fileIndex])

    #sampleLabel = audioFile.shape[1]*[lanIndex + 1]
    #padRegionLabel = (maxDim - audioFile.shape[1])*[0]
    #sampleLabel.extend(padRegionLabel)
    #outputLabelList.append(np.array(sampleLabel))
    outputLabelList.append(lanIndex + 1)

  padInfoMatrix.append(np.array(lanPadInfoList))
  outputLabelMatrix.append(np.array(outputLabelList))
  featureMatrix.append(np.array(featureList))

padInfoMatrix = np.array(padInfoMatrix)
outputLabelMatrix = np.array(outputLabelMatrix)
featureMatrix = np.array(featureMatrix)

np.save('outputLabels.npy', outputLabelMatrix, allow_pickle = True)
np.save('padInformationMatrix.npy', padInfoMatrix, allow_pickle = True)
np.save('featureMatrix.npy', featureMatrix, allow_pickle = True)

print(outputLabelMatrix.shape, featureMatrix.shape, padInfoMatrix.shape)

os.chdir('c:\\Users\\Jack\\Desktop\\Thesis\\code\\benchmark\\data\\en')
audioFile = librosa.load(os.listdir()[0])[0]
mfcc = librosa.feature.mfcc(audioFile, samplingRate, n_mfcc = 13)
delta = librosa.feature.delta(mfcc, order = 1)
deltaDelta = librosa.feature.delta(mfcc, order = 2)
plt.figure(figsize = (10, 6))
specshow(mfcc, cmap = 'inferno', x_axis = 'time')
plt.tight_layout()
print(outputLabelMatrix[0][999])