import os
import json
import math
import time
import torch
import librosa
import numpy as np
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor


def pad(segment, new_segment_length):
  print(f"segment length = {len(segment)}")
  new_segment = np.pad(segment, (0, new_segment_length - len(segment)))
  print(f"new_segment length = {len(new_segment)}")
  return new_segment

if __name__ == "__main__":
  start_time = time.time()
  config_file = open("c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\segmentation_config.json")
  config = json.load(config_file)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # setup
  if config["LID_system"] == "sb":
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir=config["speechbrain_model_path"], run_opts={"device":device})

  elif config["LID_system"] == "hu":
    model = HubertForSequenceClassification.from_pretrained(config["hubert_model_type"])
    model.classifier = nn.Linear(256, config["num_languages"])
    model.load_state_dict(torch.load(config["hubert_model_path"]))
    model = model.to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_extractor"])


  segment_length = config["segment_length"]
  sampling_rate = config["sampling_rate"]
  samples_per_segment = segment_length * sampling_rate



  for filename in os.listdir(config["data_dir"]):
    print(f"Predicting {filename}")
    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=config["sampling_rate"], mono=True)[0]

    num_segments = math.ceil(len(signal)/samples_per_segment)
    for i in range(0, num_segments):
      
      if i == num_segments - 1:
        segment = signal[i*samples_per_segment : len(signal)]
        segment = np.pad(segment, (0, samples_per_segment - len(segment)))
      else:
        segment = signal[i*samples_per_segment : (i+1)*samples_per_segment]
      
      segment = torch.tensor(segment)
      if config["LID_system"] == "sb":
        # make inference using speechbrain model
        segment.to(device).contiguous()
        predictions = language_id.classify_batch(segment)

      elif config["LID_system"] == "hu":
        # make inference using fine-tuned HuBERT model 
        inputs = feature_extractor(segment, sampling_rate = sampling_rate, padding = 'max_length', max_length = sampling_rate*segment_length, return_tensors = 'pt', return_attention_mask = True, truncation = True)
        inputs["input_values"] = inputs["input_values"].to(device).contiguous()
        inputs["attention_mask"] = inputs["attention_mask"].to(device).contiguous()
        predictions = model(**inputs).logits.detach()
      else:
        print("Unrecognised LID system. Exiting...")
        break
      print(f"segment {i}")
      print(predictions)


  config_file.close()
  print("--- %s seconds ---" % (time.time() - start_time))