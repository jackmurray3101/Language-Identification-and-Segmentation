import os
import json
import math
import torch
import librosa
import numpy as np
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from segmentation3 import segmentation


def pad(segment, new_segment_length):
  print(f"segment length = {len(segment)}")
  new_segment = np.pad(segment, (0, new_segment_length - len(segment)))
  print(f"new_segment length = {len(new_segment)}")
  return new_segment


if __name__ == "__main__":
  config_file = open("c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\segmentation_config2.json")
  config = json.load(config_file)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # setup
  if config["LID_system"] == "sb":
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=config["speechbrain_model_path"], run_opts={"device":device})
    
    # as this system performs LID on 107 languages, these are just the ones we care about
    language_index_mapping = {
      "da": 17,
      "de": 18,
      "el": 19,
      "en": 20,
      "es": 22,
      "fr": 28,
      "it": 43,
      "ja": 45,
      "ko": 51,
      "nl": 68,
      "no": 70,
      "pt": 75,
      "sv": 89,
      "zh": 106
    }

  elif config["LID_system"] == "hu":
    model = HubertForSequenceClassification.from_pretrained(config["hubert_model_type"])
    model.classifier = nn.Linear(256, config["num_languages"])
    model.load_state_dict(torch.load(config["hubert_model_path"]))
    model = model.to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_extractor"])


  languages = config["languages"]
  segment_length = config["segment_length"]
  sampling_rate = config["sampling_rate"]
  hop_time = config["hop_time"]
  samples_per_segment = segment_length * sampling_rate
  samples_per_hop = hop_time * sampling_rate
  softmax = nn.Softmax(dim=1)

  for filename in os.listdir(config["data_dir"]):
    print("#########################################")
    print(f"Segmenting {filename}")
    print("#########################################")

    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=config["sampling_rate"], mono=True)[0]

    if len(signal) <= samples_per_segment:
      num_segments = 1
      signal = np.pad(signal, (0, samples_per_segment - len(signal)))
    else:
      signal = np.pad(signal, (0, samples_per_hop - (len(signal) % samples_per_hop)))

    num_segments = 1 + (len(signal) - samples_per_segment)//samples_per_hop
    all_predictions = torch.empty(num_segments, len(languages))

    for i in range(0, num_segments):
      
      segment = signal[i*samples_per_hop : i*samples_per_hop + samples_per_segment]
      segment = torch.tensor(segment)
      if config["LID_system"] == "sb":
        # make inference using speechbrain model
        segment.to(device).contiguous()
        predictions = language_id.classify_batch(segment)
        predictions = predictions[0]
        refined_predictions = torch.empty(len(languages))
        for j, language in enumerate(languages):
          refined_predictions[j] = predictions[0][language_index_mapping[language]]
        predictions = refined_predictions
      elif config["LID_system"] == "hu":
        # make inference using fine-tuned HuBERT model 
        inputs = feature_extractor(segment, sampling_rate = sampling_rate, padding = 'max_length', max_length = sampling_rate*segment_length, return_tensors = 'pt', return_attention_mask = True, truncation = True)
        inputs["input_values"] = inputs["input_values"].to(device).contiguous()
        inputs["attention_mask"] = inputs["attention_mask"].to(device).contiguous()
        predictions = model(**inputs).logits.detach()

      else:
        print("Unrecognised LID system. Exiting...")
        break

      all_predictions[i] = predictions
    # segmentation
    all_predictions = softmax(all_predictions)   
    language_sequence, segments_per_language, transitions = segmentation(all_predictions, languages)

    # print output

    print("-------------------------")
    print("----Language Sequence----")
    print("-------------------------")
    for seg, pred in language_sequence.items():
      print(f"{seg}: {pred}")

    print("-------------------------")
    print("--Segments Per Language--")
    print("-------------------------")
    segments_per_language_dict = {}
    for i in range(0, len(languages)):
      segments_per_language_dict[languages[i]] = segments_per_language[i]
    print(segments_per_language_dict)

    print("-------------------------")
    print("-------Transitions-------")
    print("-------------------------")
    if len(transitions) == 0:
      print("No language transitions occurred")
    else:
      print(*transitions, sep="\n")
  config_file.close()