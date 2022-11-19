'''
  Dataloader for VoxLingua107 data
'''

import os
import torchaudio
from torch.utils.data import Dataset
import audio_transforms as T
import librosa

class VoxLingua107(Dataset):
  
  def __init__(self, audio_dir_path, labels_file, sampling_rate, max_length, config_languages, transform=None):
    # data directory structure assumes that the labels file will be in the top level,
    # and the samples will be in subdirectories, where the name of the subdirectory
    # corresponds to the sample label
    
    self.audio_dir = audio_dir_path
    self.labels_file = labels_file
    self.sr = sampling_rate
    
    self.labels_map = {}
    for i in range(0, len(config_languages)):
      self.labels_map[config_languages[i]] = i

    # the maximum length of an audio file from the dataset (number of samples, e.g. if max was 3s @ 16kHz, ml = 48000)
    self.max_length = max_length
    
    # dictionary that maps filename to its label
    # e.g. file_label[english_sample.wav] = 'en'
    self.filename_to_label = {}
    
    # array which contains all of the filenames
    # e.g. filenames[4] = english_sample.wav
    self.filenames = []
    
    # set of IS0 2 code for language, e.g. english -> en
    self.languages = []
        
    self.transform = transform
    
    # labels file should be in format:
    # english_sample.wav en
    # german_sample.wav de
    # etc...
    labels_file_path = os.path.join(self.audio_dir, labels_file)
    for line in open(labels_file_path):
      filename, label = line.split()
      if label not in config_languages:
        continue

      if label not in self.languages:
        self.languages.append(label)
                
      self.filenames.append(filename)
      self.filename_to_label[filename] = label
      
  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    audio_sample_path = self.get_audio_sample_path(index)
    signal = librosa.load(audio_sample_path, sr=self.sr, mono=True)[0]
    label = self.get_sample_label(index)
    signal, mask = self.transform(signal)
    return signal, mask, self.labels_map[label]
  
  def get_sample_rate(self, index):
    audio_sample_path = self.get_audio_sample_path(index)
    signal, sr = torchaudio.load(audio_sample_path)
    return sr
    
  def get_sample_label(self, index):
    filename = self.filenames[index]
    return self.filename_to_label[filename]
    
  def get_audio_sample_path(self, index):
    # audio_sample_path will be in format:
    filename = self.filenames[index]
    label = self.filename_to_label[filename]
    return  os.path.join(self.audio_dir, label, filename)
  