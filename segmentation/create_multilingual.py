import math
import random
import librosa
import numpy as np
import soundfile as sf

########
class_num = 4
num_files = 6
lan1 = "en"
lan2 = "es"
lan3 = "ko"


src1_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan1}-male.wav"
src2_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan1}-female.wav"
src3_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan2}-male.wav"
src4_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan2}-female.wav"
src5_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan3}-male.wav"
src6_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\{lan3}-female.wav"
dst_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\multilingual_data\\class{class_num}\\{lan1}-{lan2}-{lan3}-{class_num}.wav"
label_filepath = f"c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\multilingual_data\\class{class_num}\\{lan1}-{lan2}-{lan3}-{class_num}.txt"
sr = 16000

if class_num == 1:
  min_time = 20
  max_time = 60
elif class_num == 2:
  min_time = 60
  max_time = 180
elif class_num == 3:
  min_time = 180
  max_time = 240
else:
  min_time = 5
  max_time = 20

min_samples = min_time * sr
max_samples = max_time * sr

signal1 = librosa.load(src1_filepath, sr=sr, mono=True)[0]
signal2 = librosa.load(src2_filepath, sr=sr, mono=True)[0]
signal3 = librosa.load(src3_filepath, sr=sr, mono=True)[0]
signal4 = librosa.load(src4_filepath, sr=sr, mono=True)[0]
signal5 = librosa.load(src5_filepath, sr=sr, mono=True)[0]
signal6 = librosa.load(src6_filepath, sr=sr, mono=True)[0]

signals = [signal1, signal2, signal3, signal4, signal5, signal6]
labels = [lan1, lan1, lan2, lan2, lan3, lan3]

##############

current_indexes = np.zeros(num_files, int)
multilingual_speech = []
transitions = []

prev_signal = -1
while True:
  curr_signal = random.randrange(num_files)
  while curr_signal == prev_signal and len(signals) > 1:
    curr_signal = random.randrange(num_files)

  if len(signals[curr_signal]) - current_indexes[curr_signal] < max_samples:
    # we have made it to the end of this signal
    num_files -= 1
    del signals[curr_signal]
    del labels[curr_signal]
    current_indexes = np.delete(current_indexes, curr_signal)
    if len(signals) == 0:
      break
    else:
      continue

  language_length = random.randrange(min_samples, max_samples)
  transitions.append((len(multilingual_speech), round(len(multilingual_speech)/sr, 2),  labels[curr_signal]))
  multilingual_speech = np.append(multilingual_speech, signals[curr_signal][current_indexes[curr_signal] : current_indexes[curr_signal] + language_length])

  current_indexes[curr_signal] += language_length
  prev_signal = curr_signal

sf.write(dst_filepath, multilingual_speech, sr)
f = open(label_filepath, "w+")
for t in transitions:
  f.write(f"{t[0]},{t[1]},{t[2]}\n")
f.close()