import numpy as np

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


def segmentation(all_predictions, languages):

  segments_per_language = np.zeros(len(languages))
  language_sequence = {}
  transitions = []
  curr_language = 0
  prev_language = 0
  threshold = 0.25
  edge_buffer = 5

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
      left = 0.3 * all_predictions[i-3] + 0.3 * all_predictions[i-2] + 0.4 * all_predictions[i-1]
      right = 0.4 * all_predictions[i] + 0.3 * all_predictions[i+1] + 0.3 * all_predictions[i+2]
      dp = np.dot(left, right)
      print(f"dp = {dp}")
      if dp < threshold:
        # transition may have occured
        print("Below threshold")
        prev_language = curr_language
        curr_language = right.argmax().item()
        if curr_language == prev_language:
          print("But no change")
        else:
          print("tranisiton occured")
          transitions.append(f"Transition from {languages[prev_language]} to {languages[curr_language]} between segment {i-1} and {i}")

      language_sequence[f"Segment {i}"] = languages[curr_language]
      segments_per_language[curr_language] += 1

  return language_sequence, segments_per_language, transitions