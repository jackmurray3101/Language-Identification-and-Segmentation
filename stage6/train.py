import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Data2VecAudioForSequenceClassification
from torchvision import transforms
import audio_transforms as T
from data import VoxLingua107
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import json
import sys

def validate_network(model, valloader):
  print("Validating")
  model.eval()
  total_val_signals = 0
  total_val_correct = 0
  with torch.no_grad():
    for batch in valloader:
      signals, mask, labels = batch
      signals = signals.to(device).contiguous()
      mask = mask.to(device).contiguous()
      labels = labels.to(device).contiguous()
      inputs = {}
      inputs["input_values"] = signals.float()
      inputs["attention_mask"] = mask.long()
      predictions = model(**inputs).logits
      output = predictions.argmax(dim=1)
      total_val_signals += labels.size(0)
      total_val_correct += output.eq(labels).sum().item()
  model_accuracy = (total_val_correct / total_val_signals) * 100
  print("Val set size: {0}, validation accuracy {1:.2f}%".format(total_val_signals, model_accuracy))
  model.train()

def test_network(model, testloader, num_languages, languages):
  print("Testing")
  model.eval()
  total_test_signals = 0
  total_test_correct = 0
  confusion_matrix = torch.tensor(np.zeros((num_languages, num_languages)), dtype=torch.int32)
  confusion_matrix.to(device).contiguous()
  with torch.no_grad():
    for batch in testloader:
      signals, mask, labels = batch
      signals = signals.to(device).contiguous()
      mask = mask.to(device).contiguous()
      labels = labels.to(device).contiguous()
      inputs = {}
      inputs["input_values"] = signals.float()
      inputs["attention_mask"] = mask.long()
      predictions = model(**inputs).logits
      output = predictions.argmax(dim=1)
      total_test_signals += labels.size(0)
      total_test_correct += output.eq(labels).sum().item()
      for i in range (0, len(labels)):
        confusion_matrix[labels[i]][output[i]] += 1

  print(" ".join(languages))
  print(confusion_matrix.numpy())

  model_accuracy = (total_test_correct / total_test_signals) * 100
  print("Test set size: {0}, Test accuracy {1:.2f}%".format(total_test_signals, model_accuracy))
  model.train()


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(device)

  # load parameters
  config_file = open(sys.argv[1])
  config = json.load(config_file)

  sr = config["sampling_rate"]        
  max_length = config["sample_duration"]

  train_dir = config["train_dir"]
  val_dir = config["val_dir"]
  test_dir = config["test_dir"]
  
  epochs = config["epochs"]
  batch_size = config["batch_size"]
  num_languages = config["num_languages"]


  # Data Augmentation
  random_transforms = transforms.Compose([
    T.Extractor(config["extractor_path"], max_length=max_length, sampling_rate=sr)
  ])
  
  train_data = VoxLingua107(train_dir, "labels.txt", sr, max_length, config["languages"], transform=random_transforms)
  val_data = VoxLingua107(val_dir, "labels.txt", sr, max_length, config["languages"], transform=random_transforms)
  test_data = VoxLingua107(test_dir, "labels.txt", sr, max_length, config["languages"], transform=random_transforms)
  model =Data2VecAudioForSequenceClassification.from_pretrained(config["model_path"])

  # Define training parameters  
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
  scheduler = ReduceLROnPlateau(optimizer, "min", patience=config["patience"])


  multiGPU = False
  if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs Used")
    model = nn.DataParallel(model)
    multiGPU = True
  
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
  valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
  testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
  
  if (multiGPU):
    # adjust final FC layer for LID
    model.module.classifier = nn.Linear(256, num_languages)
    
    # Freeze other layers
    for param in model.module.data2vec_audio.feature_extractor.parameters():
      param.requires_grad = False
      
    for param in model.module.data2vec_audio.encoder.parameters():
      # this will set all of the transformer grads to false as well
      param.requires_grad = False
  else:
    # adjust final FC layer for LID
    model.classifier = nn.Linear(256, num_languages)
    
    # Freeze other layers
    for param in model.data2vec_audio.feature_extractor.parameters():
      param.requires_grad = False
      
    for param in model.data2vec_audio.encoder.parameters():
      # this will set all of the transformer grads to false as well
      param.requires_grad = False
  
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("Trainable Parameters : " + str(params))
  
  print("Start training...")
  model.train()
  model.to(device)
  for epoch in range(epochs):
    total_loss = 0
    total_signals = 0
    total_correct = 0
    
    for batch in trainloader:
      signals, mask, labels = batch
      signals = signals.to(device).contiguous()
      mask = mask.to(device).contiguous()
      labels = labels.to(device).contiguous()
      
      inputs = {}
      inputs["input_values"] = signals.float()
      inputs["attention_mask"] = mask.long()
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
      print(f"epoch {epoch}, loss: {loss.item():.2f}, train {model_accuracy:.2f}%")
      
    print("Epoch completed!")
    print(f"epoch {epoch}, total signals {total_signals}, average_loss_per_batch: {total_loss/len(trainloader):2f}, train {model_accuracy:.2f}%")
    validate_network(model, valloader)
    scheduler.step(total_loss/len(trainloader))

  print("Training Completed!")
  test_network(model, testloader, num_languages, config["languages"])
