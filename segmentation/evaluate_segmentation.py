'''
  Evaluates the provided segmentaion system.
  Select appropriate config file and segmentation system in imports
'''

import os
import json
import math
import torch
import librosa
import numpy as np
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
from segmentation3 import segmentation
from speechbrain.pretrained import EncoderClassifier
from pyannote.core import Timeline, Segment, Annotation, notebook
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationRecall, SegmentationPurity, SegmentationCoverage


# Returns an array containing the following metrics:
# [
#   frame accuracy,
#   segment accuracy,
#   precison with 2s tolerace,
#   precison with 5s tolerace,
#   precison with 10s tolerace,
#   recall with 2s tolerace,
#   recall with 5s tolerace,
#   recall with 10s tolerace,
#   purity,
#   coverage
# ]
def compute_metrics(all_actual_transitions, predicted_transitions, file_length, sampling_rate, samples_per_segment, samples_per_hop, plots_dir, exp_num, pure_filename):
  file_duration = file_length/sampling_rate
  metrics = []
  # labels in dataset contain transitions between speakers of the same language
  # remove these so only language transitions are present
  actual_transitions = []
  prev_language = all_actual_transitions[0].split(",")[2]
  actual_transitions.append(all_actual_transitions[0])
  for i in range(1, len(all_actual_transitions)):
    curr_language = all_actual_transitions[i].split(",")[2]
    if prev_language != curr_language:
      actual_transitions.append(all_actual_transitions[i])
    prev_language = curr_language

  if (not len(actual_transitions) or not len(predicted_transitions)):
    print("Error: Length of transition vector 0, unable to compute metrics")
    exit(1)

  # Calculate frame Accuracy
  num_correct = 0
  ref_language = actual_transitions[0].split(",")[2]
  hyp_language = predicted_transitions[0].split(",")[2]
  hyp_index = 1
  ref_index = 1
  for i in range(file_length):
    if ref_index < len(actual_transitions) and int(actual_transitions[ref_index].split(",")[0]) == i:
      ref_language = actual_transitions[ref_index].split(",")[2]
      ref_index += 1

    if hyp_index < len(predicted_transitions) and int(predicted_transitions[hyp_index].split(",")[0]) == i:
      hyp_language = predicted_transitions[hyp_index].split(",")[2]
      hyp_index += 1

    if ref_language == hyp_language:
      num_correct += 1
    
  frame_accuracy = 100 * num_correct/file_length  


  # Calculate Segment Accuracy

  num_correct = 0
  ref_language = actual_transitions[0].split(",")[2]
  hyp_language = predicted_transitions[0].split(",")[2]
  hyp_index = 1
  ref_index = 1
  num_segments = 1 + (file_length - samples_per_segment)//samples_per_hop
  for i in range(num_segments):
    start_sample = i * samples_per_hop
    end_sample = start_sample + samples_per_segment
    halfway_sample = math.floor((end_sample - start_sample)/2) + start_sample
    ref_majority_lan = ref_language
    hyp_majority_lan = hyp_language
    if ref_index < len(actual_transitions) and  hyp_index < len(predicted_transitions) and end_sample >= int(actual_transitions[ref_index].split(",")[0]) and end_sample >= int(predicted_transitions[hyp_index].split(",")[0]):
      # transition in both reference and hypothesis during this segment
      if int(actual_transitions[ref_index].split(",")[0]) < halfway_sample:
        ref_majority_lan = actual_transitions[ref_index].split(",")[2]
      if int(predicted_transitions[hyp_index].split(",")[0]) < halfway_sample:
        hyp_majority_lan = predicted_transitions[hyp_index].split(",")[2]

      ref_language = actual_transitions[ref_index].split(",")[2]
      ref_index += 1
      hyp_language = predicted_transitions[hyp_index].split(",")[2]
      hyp_index += 1
    elif ref_index < len(actual_transitions) and end_sample >= int(actual_transitions[ref_index].split(",")[0]):
      # transition in reference during this segment
      if int(actual_transitions[ref_index].split(",")[0]) < halfway_sample:
        ref_majority_lan = actual_transitions[ref_index].split(",")[2]

      ref_language = actual_transitions[ref_index].split(",")[2]
      ref_index += 1
    # If no transitions
    elif  hyp_index < len(predicted_transitions) and end_sample >= int(predicted_transitions[hyp_index].split(",")[0]):
      # transition in hypothesis during this segment
      if int(predicted_transitions[hyp_index].split(",")[0]) < halfway_sample:
        hyp_majority_lan = predicted_transitions[hyp_index].split(",")[2]
      
      hyp_language = predicted_transitions[hyp_index].split(",")[2]
      hyp_index += 1
    
    if ref_majority_lan == hyp_majority_lan:
        num_correct += 1

  segment_accuracy = 100 * num_correct/num_segments 
  # Calculate Precision and Recall, Coverage and Purity
  reference = Timeline()
  hypothesis = Timeline()
  reference_annotation = Annotation()
  hypothesis_annotation = Annotation()
  prev_transition = actual_transitions[0]
  frame_num, time, language = prev_transition.split(",")
  for i in range(1, len(actual_transitions)):
    prev_time = time
    prev_language = language
    frame_num, time, language = actual_transitions[i].split(",")
    reference.add(Segment(float(prev_time), float(time)))
    reference_annotation[Segment(float(prev_time), float(time))] = prev_language
  reference.add(Segment(float(time), float(file_duration)))
  reference_annotation[Segment(float(time), float(file_duration))] = language

  prev_transition = predicted_transitions[0]
  frame_num, time, language = prev_transition.split(",")
  for i in range(1, len(predicted_transitions)):
    prev_time = time
    prev_language = language
    frame_num, time, language = predicted_transitions[i].split(",")
    hypothesis.add(Segment(float(prev_time), float(time)))
    hypothesis_annotation[Segment(float(prev_time), float(time))] = prev_language
  hypothesis.add(Segment(float(time), float(file_duration)))
  hypothesis_annotation[Segment(float(time), float(file_duration))] = language

  plots_dir = os.path.join(plots_dir, f"exp{exp_num}")
  try:
    os.mkdir(plots_dir)
  except:
    pass
  plot_name = f"exp{exp_num}_{pure_filename}_timeline"
  plot_path = os.path.join(plots_dir, plot_name)
  notebook.width = 10
  fig = plt.figure(figsize = (10, 5))
  plt.rcParams['figure.figsize'] = (notebook.width, 3)

  notebook.crop = Segment(0, 900)

  # plot reference
  notebook.plot_annotation(reference_annotation, legend=True, time=True)
  plt.gca().text(0.6, 0.15, 'reference', fontsize=16)
  red_plotname = f"{plot_path}_ref"
  plt.savefig(red_plotname, dpi=250)
  fig = plt.figure(figsize = (10, 5))
  notebook.plot_annotation(hypothesis_annotation, legend=True, time=True)
  plt.gca().text(0.6, 0.15, 'hypothesis with HuBERT LID system', fontsize=16)
  hyp_plotname = f"{plot_path}_hyp"
  plt.savefig(hyp_plotname, dpi=250)

  precision2 = SegmentationPrecision(tolerance=2)
  recall2 = SegmentationRecall(tolerance=2)
  precision5 = SegmentationPrecision(tolerance=5)
  recall5 = SegmentationRecall(tolerance=5)
  precision10 = SegmentationPrecision(tolerance=10)
  recall10 = SegmentationRecall(tolerance=10)
  coverage = SegmentationCoverage()
  purity = SegmentationPurity()

  prec2 = 100 * precision2(reference, hypothesis)
  rec2 = 100 * recall2(reference, hypothesis)
  prec5 = 100 * precision5(reference, hypothesis)
  rec5 = 100 * recall5(reference, hypothesis)
  prec10 = 100 * precision10(reference, hypothesis)
  rec10 = 100 * recall10(reference, hypothesis)
  cov = 100 * coverage(reference_annotation, hypothesis_annotation)
  pur = 100 * purity(reference_annotation, hypothesis_annotation)

  metrics.append(frame_accuracy)
  metrics.append(segment_accuracy)
  metrics.append(prec2)
  metrics.append(prec5)
  metrics.append(prec10)
  metrics.append(rec2)
  metrics.append(rec5)
  metrics.append(rec10)
  metrics.append(pur)
  metrics.append(cov)
  return metrics

def plot_results(results, plots_dir, exp_num, class_num):
  filenames = []
  frame_accuracy = []
  segment_accuracy = []
  prec2 = []
  prec5 = []
  prec10 = []
  rec2 = []
  rec5 = []
  rec10 = []
  pur = []
  cov = []
  print("filenames,frame_accuracy,segment_accuracy,prec2,prec5,prec10,rec2,rec5,rec10,purity,coverage")
  for filename in results.keys():
    filenames.append(filename)
    frame_accuracy.append(results[filename][0])
    segment_accuracy.append(results[filename][1])
    prec2.append(results[filename][2])
    prec5.append(results[filename][3])
    prec10.append(results[filename][4])
    rec2.append(results[filename][5])
    rec5.append(results[filename][6])
    rec10.append(results[filename][7])
    pur.append(results[filename][8])
    cov.append(results[filename][9])
    print(filename, end="")
    for metric in results[filename]:
      print(f",{metric}", end="")
    print()

  # plots
  plots_dir = os.path.join(plots_dir, f"exp{exp_num}")
  try:
    os.mkdir(plots_dir)
  except:
    pass
  # Frame accuracy
  fig = plt.figure(figsize = (10, 5))
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, frame_accuracy)
  plt.title(f"Class {class_num} frame accuracy")
  plt.xlabel("Filenames")
  plt.ylabel("Accuracy (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_frame_accuracy"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # Segment accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, segment_accuracy)
  plt.title(f"Class {class_num} segment accuracy")
  plt.xlabel("Filenames")
  plt.ylabel("Accuracy (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_segment_accuracy"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # prec2 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, prec2)
  plt.title(f"Class {class_num} precision with 2s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Precision (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_prec2"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # prec5 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, prec5)
  plt.title(f"Class {class_num} precision with 5s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Precision (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_prec5"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # prec10 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, prec10)
  plt.title(f"Class {class_num} precision with 10s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Precision (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_prec10"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # rec2 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, rec2)
  plt.title(f"Class {class_num} recall with 2s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Recall (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_rec2"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # rec2 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, rec5)
  plt.title(f"Class {class_num} recall with 5s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Recall (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_rec5"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # rec10 accuracy
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, rec10)
  plt.title(f"Class {class_num} recall with 10s tolerance")
  plt.xlabel("Filenames")
  plt.ylabel("Recall (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_rec10"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # purity
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, pur)
  plt.title(f"Class {class_num} purity")
  plt.xlabel("Filenames")
  plt.ylabel("Purity (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_purity"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)

  # coverage
  fig = plt.figure(figsize = (10, 5))
  plt.bar(filenames, cov)
  plt.title(f"Class {class_num} covergae")
  plt.xlabel("Filenames")
  plt.ylabel("Coverage (%)")
  plt.ylim([0, 100])
  plot_name = f"exp{exp_num}_coverage"
  plot_path = os.path.join(plots_dir, plot_name)
  plt.savefig(plot_path, dpi=250)


  # compute averages
  num_files = len(filenames)

  frame_avg = sum(frame_accuracy)/num_files
  seg_avg = sum(segment_accuracy)/num_files
  prec2_avg = sum(prec2)/num_files
  prec5_avg = sum(prec5)/num_files
  prec10_avg = sum(prec10)/num_files
  rec2_avg = sum(rec2)/num_files
  rec5_avg = sum(rec5)/num_files
  rec10_avg = sum(rec10)/num_files
  pur_avg = sum(pur)/num_files
  cov_avg = sum(cov)/num_files
  print(f"average,{frame_avg},{seg_avg},{prec2_avg},{prec5_avg},{prec10_avg},{rec2_avg},{rec5_avg},{rec10_avg},{pur_avg},{cov_avg}")


if __name__ == "__main__":
  config_file = open("c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\segmentation_config2.json")
  config = json.load(config_file)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  results = {}


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
  edge_extension_duration = config["edge_extension_duration"]
  num_edge_extensions = config["num_edge_extensions"]
  samples_per_segment = segment_length * sampling_rate
  samples_per_hop = hop_time * sampling_rate
  log_softmax = nn.LogSoftmax(dim=1)
  plots_dir = config["plots_dir"]
  exp_num = config["experiment_number"]
  class_num = config["class_number"]

  if segment_length % hop_time != 0:
    print("Error, segment length should be a multiple of hop time")
    exit(1)

  for filename in os.listdir(config["data_dir"]):
    label_name = filename.split(".")
    label_name = label_name[0]
    pure_filename = label_name
    label_name = label_name + ".txt"
    label_filepath = os.path.join(config["labels_dir"], label_name)

    f = open(label_filepath, "r")
    actual_transitions = []
    for line in f:
      actual_transitions.append(line.strip())
    f.close()
    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=sampling_rate, mono=True)[0]

    if len(signal) <= samples_per_segment:
      num_segments = 1
      signal = np.pad(signal, (0, samples_per_segment - len(signal)))
    else:
      if len(signal) % samples_per_hop:
        signal = np.pad(signal, (0, samples_per_hop - (len(signal) % samples_per_hop)))

    edge_samples = edge_extension_duration * sampling_rate
    start = signal[0: edge_samples]
    end = signal[len(signal) - edge_samples : len(signal)]

    original_file_length = len(signal)

    # extend start and end to edge of signal to deal with edge cases
    extended_signal = np.zeros(len(signal) + 2 * edge_samples * num_edge_extensions)
    real_start = edge_samples * num_edge_extensions
    real_end = real_start + len(signal)
    extended_signal[real_start : real_end] = signal

    for i in range(num_edge_extensions):
      extended_signal[i * edge_samples : edge_samples * (i+1)] = start
      extended_signal[i * edge_samples + real_end : (i+1) * edge_samples + real_end] = end
    signal = extended_signal
    
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
    offset_predicted_transitions = segmentation(all_predictions, languages, segment_length, hop_time, sampling_rate)
    
    # remove offset from transition due to prepended data on edges
    predicted_transitions = []
    frames_appended = edge_samples * num_edge_extensions
    time_appended = frames_appended/sampling_rate
    for transition in offset_predicted_transitions:
      fields = transition.split(",")
      language = fields[2]
      if int(fields[0]) == 0:
        frame = fields[0]
        time = fields[1]
      else:
        frame = int(fields[0]) - frames_appended
        time = float(fields[1]) - time_appended
      # check transition occured within range of original signal, not in prepended/appended edges
      if int(frame) >= 0 and int(frame) < original_file_length:
        predicted_transitions.append(f"{frame},{time},{language}")

    metrics = compute_metrics(actual_transitions, predicted_transitions, original_file_length, sampling_rate, samples_per_segment, samples_per_hop, plots_dir, exp_num, pure_filename)
    results[pure_filename] = metrics
  config_file.close()

plot_results(results, plots_dir, exp_num, class_num)
  