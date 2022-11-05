import numpy as np
import torch

'''
  Find language transitions given a sequence of language probabilities
  
  Methodology:
    - dot product


  Input:
    all_predicitons
      2D Tensor with shape (number of segments) x (number of languages),
      containing the independent probiblity that speech in the given segment was
      in the corresponding language
    languages
      contains the iso 639-1 codes of the languages that the indexes correspond to

  Output:
    language sequence
      contains the iso code of the language contained in each segment
    segments_per_language
      for each language, how many segements contained this language
    transitions
      array of strings containing information on where language transitions occurred
'''


def segmentation(all_predictions, languages, segment_length, hop_time, sr):

  segments_per_language = np.zeros(len(languages))
  language_sequence = {}
  transitions = []
  curr_language = 0
  prev_language = 0
  threshold = 0.0001
  edge_buffer = 4
  window_size = 3
  offset = int(segment_length/hop_time)
  prev_dp = 1

  if window_size > edge_buffer:
    print("Error: window_size should not be greater than edge_buffer. Some segments are being missed.")
    exit(1)

  # find initial language
  # look at average of first (edge_buffer) segments to guess initial language
  log_probs = all_predictions[0]
  for j in range(1, edge_buffer):
    log_probs += all_predictions[j]
  curr_language = log_probs.argmax().item()
  prev_language = curr_language
  transitions.append(f"0,0.0,{languages[curr_language]}")

  for i in range(window_size - 1, len(all_predictions) - window_size - offset + 1):
   
    # add log probabilities
    left = all_predictions[i]
    right = all_predictions[i+offset]
    for counter in range(1, window_size):
      left = left + all_predictions[i - counter]
      right = right + all_predictions[i + offset + counter]

    # compute dot product by adding log vectors, converting to linear, then summing vectors
    addition = left + right
    linear_probs = torch.exp(addition)
    dp = sum(linear_probs)
    print("{0:.10f}".format(dp.item()))
    if dp < threshold:
      # transition may have occured
      print("Below threshold")

      # if dp has started increasing again, than the last transition occured last time (i-1)
      if dp > prev_dp:
        prev_language = curr_language
        curr_language = right.argmax().item()
        if curr_language == prev_language:
          print("But no change")
        else:
          print("tranisiton occured")
          time = segment_length + (i-1) * hop_time
          frame = time * sr
          transitions.append(f"{frame},{time},{languages[curr_language]}")

    language_sequence[f"Segment {i}"] = languages[curr_language]
    segments_per_language[curr_language] += 1
    prev_dp = dp
  return language_sequence, segments_per_language, transitions


'''
 if i == edge_buffer:
      # look at average of first 3 segments to guess initial language
      log_probs = all_predictions[0]
      for j in range(1, edge_buffer):
        log_probs += all_predictions[j]
      for j in range(edge_buffer):
        language_sequence[f"Segment {j}"] = languages[log_probs.argmax().item()]
      segments_per_language[log_probs.argmax().item()] += edge_buffer
      curr_language = log_probs.argmax().item()
      prev_language = curr_language

    elif i == len(all_predictions) - edge_buffer:
      # look at average of last 3 segments to determine final language
      log_probs = all_predictions[i]
      for j in range(i + 1, i + edge_buffer):
        log_probs += all_predictions[j]
      for j in range(i, i + edge_buffer):
        language_sequence[f"Segment {j}"] = languages[log_probs.argmax().item()]
      segments_per_language[log_probs.argmax().item()] += edge_buffer
'''