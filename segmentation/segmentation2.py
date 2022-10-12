import numpy as np

'''
  Find language transitions given a sequence of language probabilities
  
  Methodology:
    - sliding window


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

  for i, segment_predictions in enumerate(all_predictions):
    avg = segment_predictions
    if i == 0:
      # first segment
      avg = 0.7 * segment_predictions + 0.3 * all_predictions[i+1]

    elif i + 1 == len(all_predictions):
      # last segment 
      avg = 0.3 * all_predictions[i-1] + 0.7 * segment_predictions

    else:
      avg = 0.25 * all_predictions[i-1] + 0.5 * segment_predictions + 0.25 * all_predictions[i+1]

    language_sequence[f"Segment {i}"] = languages[avg.argmax().item()]
    segments_per_language[avg.argmax().item()] += 1
    
    prev_language = curr_language
    curr_language = languages[avg.argmax().item()]
    if i:
      if curr_language != prev_language:
        # a transition has occurred
        transitions.append(f"Transition from {prev_language} to {curr_language} between segment {i-1} and {i}")

  return language_sequence, segments_per_language, transitions