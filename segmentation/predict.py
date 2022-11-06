import os
import json
import math
import torch
import librosa
import numpy as np
import torch.nn as nn
from segmentation3 import segmentation
from pyannote.core import Timeline, Segment, Annotation
from speechbrain.pretrained import EncoderClassifier
from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationRecall, SegmentationPurity, SegmentationCoverage
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def compute_metrics(actual_transitions, predicted_transitions, file_length):
  reference = Timeline()
  hypothesis = Timeline()
  reference_annotaton = Annotation()
  hypothesis_annotaton = Annotation()

  prev_transition = actual_transitions[0]
  frame_num, time, language = prev_transition.split(",")
  for i in range(1, len(actual_transitions)):
    prev_time = time
    prev_language = language
    frame_num, time, language = actual_transitions[i].split(",")
    if prev_language != language:
      reference.add(Segment(float(prev_time), float(time)))
      reference_annotaton[Segment(float(prev_time), float(time))] = language
  reference.add(Segment(float(time), float(file_length)))

  prev_transition = predicted_transitions[0]
  frame_num, time, language = prev_transition.split(",")
  for i in range(1, len(predicted_transitions)):
    prev_time = time
    frame_num, time, language = predicted_transitions[i].split(",")
    hypothesis.add(Segment(float(prev_time), float(time)))
    hypothesis_annotaton[Segment(float(prev_time), float(time))] = language
  hypothesis.add(Segment(float(time), float(file_length)))

  precision2 = SegmentationPrecision(tolerance=2)
  recall2 = SegmentationRecall(tolerance=2)
  precision5 = SegmentationPrecision(tolerance=5)
  recall5 = SegmentationRecall(tolerance=5)
  precision10 = SegmentationPrecision(tolerance=10)
  recall10 = SegmentationRecall(tolerance=10)
  coverage = SegmentationCoverage()
  purity = SegmentationPurity()

  prec2 = precision2(reference, hypothesis)
  rec2 = recall2(reference, hypothesis)
  prec5 = precision5(reference, hypothesis)
  rec5 = recall5(reference, hypothesis)
  prec10 = precision10(reference, hypothesis)
  rec10 = recall10(reference, hypothesis)
  cov = coverage(reference_annotaton, hypothesis_annotaton)
  pur = purity(reference_annotaton, hypothesis_annotaton)


  print("Reference:")
  print(reference)
  print("Hypothesis:")
  print(hypothesis)

  print("Precision with tolerance = 2s")
  print(prec2)
  print("Precision with tolerance = 5s")
  print(prec5)
  print("Precision with tolerance = 10s")
  print(prec10)

  print("Recall with tolerance = 2s")
  print(rec2)
  print("Recall with tolerance = 5s")
  print(rec5)
  print("Recall with tolerance = 10s")
  print(rec10)

  print("Purity")
  print(pur)

  print("Coverage")
  print(cov)


if __name__ == "__main__":
  config_file = open("c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\segmentation_config4.json")
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
  log_softmax = nn.LogSoftmax(dim=1)

  if segment_length % hop_time != 0:
    print("Error, segment length should be a multiple of hop time")
    exit(1)


  for filename in os.listdir(config["data_dir"]):
    print("#########################################")
    print(f"Segmenting {filename}")
    print("#########################################")

    label_name = filename.split(".")
    label_name = label_name[0]
    label_name = label_name + ".txt"
    label_filepath = os.path.join(config["labels_dir"], label_name)

    f = open(label_filepath, "r")
    actual_transitions = []
    for line in f:
      actual_transitions.append(line.strip())

    print("actual transitions:")
    print(actual_transitions)

    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=sampling_rate, mono=True)[0]

    if len(signal) <= samples_per_segment:
      num_segments = 1
      signal = np.pad(signal, (0, samples_per_segment - len(signal)))
    else:
      if len(signal) % samples_per_hop:
        signal = np.pad(signal, (0, samples_per_hop - (len(signal) % samples_per_hop)))

    file_length = len(signal)/sampling_rate
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
    all_predictions = log_softmax(all_predictions)
    predicted_transitions = segmentation(all_predictions, languages, segment_length, hop_time, sampling_rate)
    compute_metrics(actual_transitions, predicted_transitions, file_length)

    print("-------------------------")
    print("-------Transitions-------")
    print("-------------------------")
    if len(predicted_transitions) == 0:
      print("No language transitions occurred")
    else:
      print(*predicted_transitions, sep="\n")
  config_file.close()