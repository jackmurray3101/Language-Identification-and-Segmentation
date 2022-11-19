import os
import torchaudio
from torch.utils.data import Dataset
import audio_transforms as T
import librosa

class VoxLingua107(Dataset):
  
  def __init__(self, audio_dir_path, labels_file, sampling_rate, max_length, balance=False, transform=None):
    # data directory structure assumes that the labels file will be in the top level,
    # and the samples will be in subdirectories, where the name of the subdirectory
    # corresponds to the sample label
    
    self.audio_dir = audio_dir_path
    self.labels_file = labels_file
    self.sr = sampling_rate
    
    self.labels_map = {
      'da': 0,
      'de': 1,
      'el': 2,
      'en': 3,
      'es': 4,
      'fr': 5,
      'it': 6,
      'ja': 7,
      'ko': 8,
      'nl': 9,
      'no': 10,
      'pt': 11,
      'sv': 12,
      'zh': 13
    }
    
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
    
    # used for balancing data (only balances num samples, not duration)
    self.num_samples_per_language = {}
    
    self.transform = transform
    # wav2vec 2.0 model
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    self.model =  model = bundle.get_model()  
    
    # labels file should be in format:
    # english_sample.wav en
    # german_sample.wav de
    # etc...
    labels_file_path = os.path.join(self.audio_dir, labels_file)
    for line in open(labels_file_path):
      filename, label = line.split()
      if label not in self.languages:
        self.languages.append(label)
        self.num_samples_per_language[label] = 1
      else:
        self.num_samples_per_language[label] += 1
        
      self.filenames.append(filename)
      self.filename_to_label[filename] = label
      
    if (balance):
      self.balance()

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
    # /other_dirs/thesis/data/voxlingua107/en/english_sample.wav
    filename = self.filenames[index]
    label = self.filename_to_label[filename]
    return  os.path.join(self.audio_dir, label, filename)
  
  def balance(self):
    min_samples = min(self.num_samples_per_language.values())
    tmp_num_samples_per_language = {}
    
    files_to_remove = []
    for filename in self.filenames:
      label = self.filename_to_label[filename] 
      if label not in tmp_num_samples_per_language.keys():
        tmp_num_samples_per_language[label] = 0
      else:
        tmp_num_samples_per_language[label] += 1
        
      if tmp_num_samples_per_language[label] >= min_samples:
        # remove this file from data structures
        files_to_remove.append(filename)
        self.num_samples_per_language[label] -= 1
    
    for filename in files_to_remove:
      self.filenames.remove(filename)
      self.filename_to_label.pop(filename)
    
    # check that everything is balanced correctly
    for val in self.num_samples_per_language.values():
      if val != min_samples:
        raise Exception('Encountered error while balancing data')