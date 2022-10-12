import os
import librosa

filepath = os.path.join(config["data_dir"], filename)
signal = librosa.load(filepath, sr=config["sampling_rate"], mono=True)[0]