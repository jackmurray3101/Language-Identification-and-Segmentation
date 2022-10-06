import librosa
import os
import math
import torch
import torch.nn as nn
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor


model = HubertForSequenceClassification.from_pretrained("facebook/hubert-base-ls960")
model.classifier = nn.Linear(256, 3)
model.load_state_dict(torch.load("./saved_trained_models/Hubert-3-similar-epoch40"))
model.eval()

cwd = os.getcwd()
signal_path = os.path.join(cwd, "data")
signal_path = os.path.join(signal_path, "prideandprejudice.wav")
signal = librosa.load(signal_path, sr=16000, mono=True)[0]

sampling_rate = 16000
segment_size = 20   # seconds
samples_per_segment = segment_size * 16000
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

for i in range(0, 5):
      segment = signal[i*samples_per_segment : (i+1)*samples_per_segment - 1]
      
      # if the last segment isn't large enough, the above syntax will take whatever samples
      # are left, and the w2v feature extractor will pad with zeros and return corresponding attention mask

      input_values = feature_extractor(segment, sampling_rate = sampling_rate, padding = 'max_length', max_length = sampling_rate*segment_size, return_tensors = 'pt', return_attention_mask = True, truncation = True)
      logits = model(**input_values).logits

      print(logits)
