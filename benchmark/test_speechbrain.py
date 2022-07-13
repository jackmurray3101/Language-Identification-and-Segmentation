import torch
import torchaudio
import os
from speechbrain.pretrained import EncoderClassifier


if __name__ == "__main__":
  test_data_dirname = 'data' # should be in cwd
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  languages = os.listdir(test_data_dirname)
  language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":device})
  cwd = os.getcwd()

  num_correct = 0
  num_incorrect = 0


  for language in languages:
    path = os.path.join(cwd, test_data_dirname, language)
    audio_files = os.listdir(path)
    for file_name in audio_files:

      # Make prediction. This implementation uses speechbrain
      ####################################################
      signal_path = os.path.join(path, file_name)
      signal = language_id.load_audio(signal_path)
      prediction = language_id.classify_batch(signal)
      code = prediction[3] # e.g. code = ['en: English']
      code = code[0].split(':')
      code = code[0] # e.g. code = 'en'
      ####################################################


      if code == language:
        num_correct += 1
      else:
        num_incorrect += 1

  accuracy = num_correct/(num_correct + num_incorrect)
  print(f'Accuracy of model on test set = {accuracy}')