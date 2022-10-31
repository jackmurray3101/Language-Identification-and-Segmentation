from os import path
from pydub import AudioSegment
import os
import librosa
import soundfile as sf


sr=16000
# files  
cwd = os.getcwd()
src_data_path = path.join(cwd, "longer_data\\da")                                                                       
src = path.join(src_data_path, "15min-danish-male.mp3")                                                                       
dst_data_path = path.join(cwd, "cleaned_data")                                                                       
dst = path.join(dst_data_path, "da-male2.wav")

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")

signal1 = librosa.load(dst, sr=sr, mono=True)[0]

#remove first 60 seconds, and whatever is left at the end
new_signal = signal1[sr * 60 : len(signal1)]
new_signal = new_signal[0 : sr * 60 * 8]

if len(new_signal) == 7680000:
  print("succesful")
sf.write(dst, new_signal, sr)