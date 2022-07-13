import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import math

class VoxLingua107(Dataset):

  def __init__(self, labels_file, audio_dir):
    self.labels = pd.read_csv(labels_file)
    self.audio_dir = audio_dir

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, index):
    audio_sample_path = self.get_audio_sample_path(index)
    signal, sr = torchaudio.load(audio_sample_path)
    return signal
  
  def get_sample_rate(self, index):
    audio_sample_path = self.get_audio_sample_path(index)
    signal, sr = torchaudio.load(audio_sample_path)
    return sr
    
  def get_audio_sample_path(self, index):
    return os.path.join(self.audio_dir, self.get_label(index))
  
  def get_label(self, index):
    return self.labels.iloc[index][0]

def main():
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(torchaudio.__version__)
  print(device)
  
  
  ##########################################
  ############   Loading Data   ############
  ##########################################
  
  language_data = []
  
  # Add information about the location of the data for each language
  # in the format [label_file_path, audio_directory_path]
  
  # English
  language_data.append(["./en_labels.csv", "./en"])

  # Add more languages here
  
  
  #########################################
  ############    SSL Model    ############
  #########################################
  
  # Import wav2vec 2.0 pre-trained model
  bundle = torchaudio.pipelines.WAV2VEC2_BASE
  sample_rate = 16000 # SR of Voxlingua dataset
  model = bundle.get_model()  
  
  ##########################################
  #############    Training    #############
  ##########################################
  
  # Do something with data
  # In future, may want to shuffle languages

  # Frame size in milliseconds
  frame_size = 25
  sample_period = 1.0/bundle.sample_rate
  num_samples_per_frame = round((frame_size/1000)/sample_period)

  for language in language_data:
    data = VoxLingua107(language[0], language[1])
    
    # Just start by working with a small amount of the data 
    for i in range(2):
      label = data.get_label(i)
      waveform = data[i]
      waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
      num_samples = waveform.size(dim=1)
      num_frames = math.floor(num_samples/num_samples_per_frame)
      for j in range(num_frames):
        frame = waveform[0, j*num_samples_per_frame:(j+1)*num_samples_per_frame]
        frame = frame.view(1, num_samples_per_frame)
        features, _ = model.extract_features(frame)


if __name__ == "__main__":
  main()