import math
import librosa
import numpy as np
import soundfile as sf

src1_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\data\\10min-english.wav"
src2_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\data\\10min-spanish.wav"
dst_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\data\\multilingual\\en-es-5min-cuts.wav"
sr = 16000
time_between_transitions = 5 # minutes 

signal1 = librosa.load(src1_filepath, sr=sr, mono=True)[0]
signal2 = librosa.load(src2_filepath, sr=sr, mono=True)[0]

print(len(signal1))
print(len(signal2))

samples_between_transitions = time_between_transitions * 60 * sr

num_transitions = 0

curr_signal = 1
if len(signal1) < len(signal2):
  num_transitions = 2 * math.floor(len(signal1)/samples_between_transitions)
else:
  curr_signal = 2
  num_transitions = 2 * math.floor(len(signal2)/samples_between_transitions)

new_signal = np.zeros(num_transitions * samples_between_transitions)

s1_index = 0
s2_index = 0
print(f"Length of output = {len(new_signal)}")

print(f"Num transitions = {num_transitions}")
for i in range(num_transitions):
  print(f"Iteration {i}")
  if curr_signal == 1:
    new_signal[i*samples_between_transitions : (i+1)*samples_between_transitions] = signal1[s1_index : s1_index + samples_between_transitions]
    s1_index += samples_between_transitions
    curr_signal = 2
  else:
    new_signal[i*samples_between_transitions : (i+1)*samples_between_transitions] = signal2[s2_index : s2_index + samples_between_transitions]
    s2_index += samples_between_transitions
    curr_signal = 1

sf.write(dst_filepath, new_signal, sr)