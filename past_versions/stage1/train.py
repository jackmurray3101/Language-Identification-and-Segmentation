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

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(torchaudio.__version__)
  print(device)
  
  # parameters
  sr = 16000        # Hz
  max_length = 6    # seconds 
  cwd = os.getcwd()
  audio_dir = os.path.join(cwd, '..', 'datasets', 'voxlingua107_subset')
  
  # Data Augmentation
  random_transforms = transforms.Compose([
    T.Extractor("superb/wav2vec2-base-superb-sid", max_length=max_length, sampling_rate=sr)
  ])
  
  data = VoxLingua107(audio_dir, 'labels.txt', sr, max_length, balance=True, transform=random_transforms)
  model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

  # Define training parameters  
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)
  epochs = 5
  batch_size = 30
  
  trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 5)
  
  # adjust final FC layer for LID
  model.classifier = nn.Linear(256, 3)
  
  # Freeze other layers
  for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False
    
  for param in model.wav2vec2.encoder.parameters():
    # this will set all of the transformer grads to false as well
    param.requires_grad = False
  
  print("Start training...")
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    total_signals = 0
    total_correct = 0
    
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
      print(f'epoch {epoch}, loss: {loss.item():.2f}, train {model_accuracy:.2f}%')
      
    print('Epoch completed!')
    print(f'epoch {epoch}, average_loss_per_batch: {total_loss/len(trainloader):2f}, train {model_accuracy:.2f}%')
    scheduler.step(total_loss/len(trainloader))