import torch
import torchaudio
import os
from speechbrain.pretrained import EncoderClassifier
import librosa
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="saved_trained_models/speechbrain", run_opts={"device":device})

cwd = os.getcwd()
signal_path = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\data\\10min-english.wav"
#signal = language_id.load_audio(signal_path)
#signal = torchaudio.load(signal_path)
signal = librosa.load(signal_path, sr=16000, mono=True)[0]
print(signal)
signal = torch.tensor(signal)


sr = 16000
seg_len = 20
segment1 = signal[0:sr*seg_len]
segment2 = signal[sr*seg_len:sr*seg_len*2]
pred = language_id.classify_batch(segment1) # can also use classify batch!
print(pred)
pred = language_id.classify_batch(segment2) # can also use classify batch!
print(pred)



