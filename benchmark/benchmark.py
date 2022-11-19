'''
  Conduct benchmark LID testing using speechbrain system
'''

import torch
import torchaudio
import os
import numpy as np
from speechbrain.pretrained import EncoderClassifier


if __name__ == "__main__":
  test_data_dir = os.path.join("..", "datasets", "voxlingua107_test")
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  #languages = os.listdir(test_data_dir)
  #languages = ["en", "es", "ko"]
  languages =  [
    "en",
    "de",
    "es",
    "fr",
    "ko",
    "zh",
    "sv",
    "no"
  ]

  try:
    languages.remove("labels.txt")
    print(("labels.txt removed"))
  except:
    print(("labels.txt NOT removed"))

  language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":device})
  cwd = os.getcwd()

  num_correct = 0
  num_incorrect = 0
  num_languages = len(languages)
  language_map = {}
  confusion_matrix = np.zeros((num_languages, num_languages + 1))

  for i, language in enumerate(languages):
    language_map[language] = i
  language_map["other"] = len(languages)

  for language in languages:
    path = os.path.join(cwd, test_data_dir, language)
    audio_files = os.listdir(path)
    for file_name in audio_files:

      # Make prediction. This implementation uses speechbrain
      ####################################################
      signal_path = os.path.join(path, file_name)
      signal = language_id.load_audio(signal_path)
      signal = signal[0:6*16000]
      prediction = language_id.classify_batch(signal)
      code = prediction[3] # e.g. code = ['en: English']
      code = code[0].split(':')
      code = code[0] # e.g. code = 'en'
      ####################################################
      print(f"Language: {language}, Prediction: {code}")

      if code == language:
        num_correct += 1
      else:
        num_incorrect += 1
      try:
        confusion_matrix[language_map[language]][language_map[code]] += 1
      except:
        confusion_matrix[language_map[language]][language_map["other"]] += 1


  accuracy = num_correct/(num_correct + num_incorrect)
  print(f'Accuracy of model on test set = {accuracy}')
  print('Confusion matrix: ')
  l_str = "  ".join(languages)
  print(f"  {l_str} other")
  print(confusion_matrix)