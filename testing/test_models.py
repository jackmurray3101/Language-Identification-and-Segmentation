import os
import sys
import json
import math
import torch
import librosa
import numpy as np
import torch.nn as nn
import audio_transforms as T
from data import VoxLingua107
from torchvision import transforms
from speechbrain.pretrained import EncoderClassifier
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

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



if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(torch.__version__)
  print(device)

  config_file = open("c:\\Users\\Jack\\Desktop\\Thesis\\code\\testing\\test_config.json")
  config = json.load(config_file)

  sr = config["sampling_rate"]        
  max_length = config["sample_duration"]

  test_dir = config["test_dir"]
  
  batch_size = config["batch_size"]
  num_languages = config["num_languages"]

  model = HubertForSequenceClassification.from_pretrained(config["hubert_model_type"])
  model.classifier = nn.Linear(256, config["num_languages"])
  model.load_state_dict(torch.load(config["hubert_model_path"]))
  model = model.to(device)
  model.eval()
  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_extractor"])

  random_transforms = transforms.Compose([
    T.Extractor(config["model_extractor"], max_length=max_length, sampling_rate=sr)
  ])

  test_data = VoxLingua107(test_dir, "labels.txt", sr, max_length, config["languages"], transform=random_transforms)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
  test_network(model, testloader, num_languages, config["languages"])