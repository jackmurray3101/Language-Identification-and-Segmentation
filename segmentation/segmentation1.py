import numpy as np

'''
  Find language transitions given a sequence of language probabilities
  
  Methodology:
    - based soley on outputs from LID system


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
    language_sequence[f"Segment {i}"] = languages[segment_predictions.argmax().item()]
    segments_per_language[segment_predictions.argmax().item()] += 1
    
    prev_language = curr_language
    curr_language = languages[segment_predictions.argmax().item()]
    if i:
      if curr_language != prev_language:
        # a transition has occurred
        transitions.append(f"Transition from {prev_language} to {curr_language} between segment {i-1} and {i}")

  return language_sequence, segments_per_language, transitions