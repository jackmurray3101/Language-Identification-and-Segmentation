import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2ForSequenceClassification
from torchvision import transforms
import audio_transforms as T
import librosa
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VoxLingua107(Dataset):

  # if balance is set True, the same number of files of each language will be read in
  # this will be equal to the smallest number of samples in one language
  def __init__(self, audio_dir_path, labels_file, max_length, balance=False, transform = None):
    # data directory structure assumes that the labels file will be in the top level,
    # and the samples will be in subdirectories, where the name of the subdirectory
    # corresponds to the sample label
    
    self.audio_dir = audio_dir_path
    self.labels_file = labels_file
    
    self.labels_map = {
      'de': 0,
      'en': 1,
      'zh': 2
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
      # if balance is true, remove some of the samples so that
      # there is the same number of samples for each language
      self.balance()

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    audio_sample_path = self.get_audio_sample_path(index)
    
    signal = librosa.load(audio_sample_path, sr = 16000, mono=True)[0]
    label = self.get_sample_label(index)
    # signal = self.pad_or_trim(signal) -> should be done during pre-processing
    
    signal, mask = self.transform(signal)
    
    return signal, mask, self.labels_map[label]
  
  def pad_or_trim(self, signal):
    # ensures the signal is of shape [1, self.max_length]
    difference = self.max_length - signal.size(1)
    if difference > 0:
      p = nn.ConstantPad1d((0, difference), 0)
      return p(signal)
    elif difference < 0:
      return torch.narrow(signal, 1, 0, self.max_length)
    else:
      return signal
  
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
  
def test_network(model, testloader):
  model.eval()
  total_val_signals = 0
  total_val_correct = 0
  with torch.no_grad():
    for batch in testloader:
      signals, mask, labels = batch
      inputs = {}
      inputs['input_values'] = signals.float()
      inputs['attention_mask'] = mask.long()
      predictions = model(**inputs).logits
      output = predictions.argmax(dim=1)
      total_val_signals += labels.size(0)
      total_val_correct += output.eq(labels).sum().item()
  model_accuracy = total_val_correct / total_val_signals * 100
  print(', {0} validation accuracy {1:.2f}%'.format(total_signals, model_accuracy))
  model.train()

if __name__ == "__main__":
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(torchaudio.__version__)
  print(device)
  
  # Create an instance of Dataset
  cwd = os.getcwd()
  audio_dir = os.path.join(cwd, '..', 'datasets', 'voxlingua107')
  max_length = 320000 # 20 seconds * 16kHz
  
  # Data Augmentation
  
  random_transforms = transforms.Compose([
    T.Extractor("superb/wav2vec2-base-superb-sid", max_length=10, sampling_rate=16000)
  ])
  data = VoxLingua107(audio_dir, 'labels.txt', max_length, balance=True, transform = random_transforms)
  
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

  epochs = 10
  batch_size = 2
  
  trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 5)
  model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
  
  # adjust final FC layer for LID
  model.classifier = nn.Linear(256, 3)
  model.classifier.requires_grad = True
  model.projector.requires_grad = True
  
  # Freeze other layers
  for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False
    
  for param in model.wav2vec2.feature_projection.parameters():
    param.requires_grad = False
    
  for param in model.wav2vec2.encoder.parameters():
    param.requires_grad = False
  
  num_transformers = 12
  for i in range(num_transformers):
    for param in model.wav2vec2.encoder.layers[i].parameters():
        param.requires_grad = False
         
  
  print("Start training...")
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    total_signals = 0
    total_correct = 0
    
    batch_num = 0
    for batch in trainloader:
      signals, mask, labels = batch
      inputs = {}
      inputs['input_values'] = signals.float()
      inputs['attention_mask'] = mask.long()
      
      predictions = model(**inputs).logits
      loss = loss_func(predictions, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      output = predictions.argmax(dim=1)
      
      total_loss += loss.item()
      total_signals += labels.size(0)
      total_correct += output.eq(labels).sum().item()
            
      model_accuracy = total_correct / total_signals * 100
      batch_num += 1
      print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
              epoch, loss.item(), signals, model_accuracy), end='')
    
    print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
              epoch, total_loss/batch_num, signals, model_accuracy), end='')
    
    scheduler.step(total_loss/len(trainloader))