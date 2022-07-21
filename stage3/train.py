import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification
from torchvision import transforms
import audio_transforms as T
from data import VoxLingua107
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def validate_network(model, valloader):
  print("Validating")
  model.eval()
  total_val_signals = 0
  total_val_correct = 0
  with torch.no_grad():
    i = 0
    for batch in valloader:
      signals, mask, labels = batch
      signals = signals.to(device).contiguous()
      mask = mask.to(device).contiguous()
      labels = labels.to(device).contiguous()
      inputs = {}
      inputs['input_values'] = signals.float()
      inputs['attention_mask'] = mask.long()
      predictions = model(**inputs).logits
      output = predictions.argmax(dim=1)
      total_val_signals += labels.size(0)
      total_val_correct += output.eq(labels).sum().item()
      i += 1
      if (i == 5): break
  model_accuracy = total_val_correct / total_val_signals * 100
  print(', {0} validation accuracy {1:.2f}%'.format(total_signals, model_accuracy))
  model.train()


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(torchaudio.__version__)
  print(device)
  
  # parameters
  sr = 16000        # Hz
  max_length = 6    # seconds 
  cwd = os.getcwd()
  train_dir = os.path.join(cwd, '..', 'datasets', 'voxlingua107_train')
  val_dir = os.path.join(cwd, '..', 'datasets', 'voxlingua107_val')
  test_dir = os.path.join(cwd, '..', 'datasets', 'voxlingua107_test')
  
  # Data Augmentation
  random_transforms = transforms.Compose([
    T.Extractor("superb/wav2vec2-base-superb-sid", max_length=max_length, sampling_rate=sr)
  ])
  
  train_data = VoxLingua107(train_dir, 'labels.txt', sr, max_length, balance=True, transform=random_transforms)
  val_data = VoxLingua107(val_dir, 'labels.txt', sr, max_length, balance=True, transform=random_transforms)
  test_data = VoxLingua107(test_dir, 'labels.txt', sr, max_length, balance=True, transform=random_transforms)
  model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

  multiGPU = False
  if torch.cuda.device_count() > 1:
    print(f'{torch.cuda.device_count()} GPUs Used')
    model = nn.DataParallel(model)
    multiGPU = True
    
  model.to(device)

  # Define training parameters  
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)
  epochs = 5
  batch_size = 2
  num_languages = 3
  
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 4)
  valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers = 2)
  # testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers = 2)
  
  if (multiGPU):
    # adjust final FC layer for LID
    model.module.classifier = nn.Linear(256, num_languages)
    
    # Freeze other layers
    for param in model.module.wav2vec2.feature_extractor.parameters():
      param.requires_grad = False
      
    for param in model.module.wav2vec2.encoder.parameters():
      # this will set all of the transformer grads to false as well
      param.requires_grad = False
  else:
    # adjust final FC layer for LID
    model.classifier = nn.Linear(256, num_languages)
    
    # Freeze other layers
    for param in model.wav2vec2.feature_extractor.parameters():
      param.requires_grad = False
      
    for param in model.wav2vec2.encoder.parameters():
      # this will set all of the transformer grads to false as well
      param.requires_grad = False
  
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print('Trainable Parameters : ' + str(params))
  
  print("Start training...")
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    total_signals = 0
    total_correct = 0
    
    batch_num = 0
    for batch in trainloader:
      signals, mask, labels = batch
      signals = signals.to(device).contiguous()
      mask = mask.to(device).contiguous()
      labels = labels.to(device).contiguous()
      
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
      print(f'epoch {epoch}, loss: {loss.item():.2f}, train {model_accuracy:.2f}%')
      
      batch_num += 1
      if (batch_num % 4 == 0):
        validate_network(model, valloader)
      
      
    print('Epoch completed!')
    print(f'epoch {epoch}, average_loss_per_batch: {total_loss/len(trainloader):2f}, train {model_accuracy:.2f}%')
    scheduler.step(total_loss/len(trainloader))