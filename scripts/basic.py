import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

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
  for language in language_data:
    data = VoxLingua107(language[0], language[1])
    
    # Just start by working with a small amount of the data 
    for i in range(5):
      print("-------------------------")
      print(f"Getting sample {i}")
      label = data.get_label(i)
      print(label)
      waveform = data[i]
      print("Waveform size before resampling")
      print(waveform.size())
      waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
      print("Waveform size after resampling")
      print(waveform.size())
      features, _ = model.extract_features(waveform)
      print("Features size:")
      print(features[0].size())
      print("Features extracted:")
      # print(features)
  

if __name__ == "__main__":
  main()