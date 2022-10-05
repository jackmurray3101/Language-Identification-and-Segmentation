import json
import librosa
import os


def speechbrain_predict(signal):
  pass

def hubert_predict(signal):
  pass




if __name__ == "__main__":
  config_file = open("segmentation_config.json")
  config = json.load(config_file)

  for filename in os.listdir(config["data_dir"]):
    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=config["sampling_rate"], mono=True)[0]
    
    # just take 10 minutes for initial testing
    signal = signal[0:9599999]

    if config["LID_system"] == "sb":
      speechbrain_predict(signal)
    elif config["LID_system"] == "hu":
      hubert_predict(signal)
    else:
      print("Unrecognised LID system. Exiting...")
      break

  config_file.close()