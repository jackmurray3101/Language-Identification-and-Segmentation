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


def segmentation(all_predictions, languages, segment_length, hop_time):

  segments_per_language = np.zeros(len(languages))
  language_sequence = {}
  transitions = []
  curr_language = 0
  prev_language = 0
  threshold = 0.25
  edge_buffer = 5
  offset = int(segment_length/hop_time)

  for i in range(edge_buffer, len(all_predictions) - edge_buffer):
    
    if i == edge_buffer:
      # look at average of first 3 segments to guess initial language
      avg = 0
      for j in range(edge_buffer):
        avg += all_predictions[j]
      avg = avg/edge_buffer
      for j in range(edge_buffer):
        language_sequence[f"Segment {j}"] = languages[avg.argmax().item()]

      segments_per_language[avg.argmax().item()] += edge_buffer
      curr_language = avg.argmax().item()
      prev_language = curr_language

    elif i == len(all_predictions) - 1 - edge_buffer:
      # look at average of last 3 segments to determine final language
      avg = 0
      for j in range(i, i + edge_buffer):
        avg += all_predictions[j]
      avg = avg/edge_buffer
      for j in range(edge_buffer):
        language_sequence[f"Segment {j}"] = languages[avg.argmax().item()]
      segments_per_language[avg.argmax().item()] += edge_buffer
    else:
      # add log probabilities
      print("Pred i")
      print(all_predictions[i])
      left = all_predictions[i] + all_predictions[i-1] + all_predictions[i-2]
      right = all_predictions[i+offset] + all_predictions[i+1+offset] + all_predictions[i+2+offset]
      print("Left")
      print(left)
      print("Right")
      print(right)
      # compute dot product by adding log vectors, converting to linear, then summing vectors
      addition = left + right
      linear_probs = torch.exp(addition)
      dp = sum(linear_probs)
      print(f"{dp}")
      if dp < threshold:
        # transition may have occured
        #print("Below threshold")
        prev_language = curr_language
        curr_language = right.argmax().item()
        if curr_language == prev_language:
         # print("But no change")
         pass
        else:
         # print("tranisiton occured")
          transitions.append(f"Transition from {languages[prev_language]} to {languages[curr_language]} between segment {i-1} and {i}")

      language_sequence[f"Segment {i}"] = languages[curr_language]
      segments_per_language[curr_language] += 1

  return language_sequence, segments_per_language, transitions