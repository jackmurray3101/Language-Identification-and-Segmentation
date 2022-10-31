import math
import librosa
import numpy as np
import soundfile as sf

src1_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\longer_data\\no\\new.wav"
src2_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\longer_data\\no\\new2.wav"
dst_filepath = "c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\longer_data\\no\\newnew.wav"

sr = 16000
signal1 = librosa.load(src1_filepath, sr=sr, mono=True)[0]
signal2 = librosa.load(src2_filepath, sr=sr, mono=True)[0]
print(f"len of s1 = {len(signal1)}")
print(f"len of s2 = {len(signal2)}")

signal1_new = signal1[2240000 : 4960000]

new_signal = np.zeros(len(signal1_new) + len(signal2))
new_signal[0 : len(signal1_new)] = signal1_new
new_signal[len(signal1_new) : len(new_signal)] = signal2

sf.write(dst_filepath, new_signal, sr)