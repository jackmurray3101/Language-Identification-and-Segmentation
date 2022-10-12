from os import path
from pydub import AudioSegment
import os

# files  
cwd = os.getcwd()
data_path = path.join(cwd, "data")                                                                       
src = path.join(data_path, "10min-spanish.mp3")                                                                       
dst = path.join(data_path, "10min-spanish.wav")

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")