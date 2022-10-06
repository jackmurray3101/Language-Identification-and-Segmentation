import json
import librosa
import os
import math
from transformers import HubertForSequenceClassification

def speechbrain_predict(signal):
  pass

def hubert_predict(signal):
  model = HubertForSequenceClassification.from_pretrained("facebook/hubert-base-ls960")
  model.load_state_dict(torch.load("./saved_trained_models/Hubert-3-similar-epoch40"))
  model.eval()


if __name__ == "__main__":
  config_file = open("segmentation_config.json")
  config = json.load(config_file)

  segment_size = 20   # seconds
  samples_per_segment = segment_size * config["sampling_rate"]

  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

  for filename in os.listdir(config["data_dir"]):
    filepath = os.path.join(config["data_dir"], filename)
    signal = librosa.load(filepath, sr=config["sampling_rate"], mono=True)[0]
    
    num_segments = math.ceil(len(signal)/samples_per_segment)
    for i in range(0, num_segments):
      segment = segment[i*samples_per_segment : (i+1)*samples_per_segment - 1]
      
      # if the last segment isn't large enough, the above syntax will take whatever samples
      # are left, and the w2v feature extractor will pad with zeros and return corresponding attention mask

      input_values = feature_extractor(segment, sampling_rate = sampling_rate, padding = 'max_length', max_length = sampling_rate*segment_size, return_tensors = 'pt', return_attention_mask = True, truncation = True)
    
    # TODO
    # Explore running inferences on batches for speed up, instead of consecutively


    if config["LID_system"] == "sb":
      speechbrain_predict(signal)
    elif config["LID_system"] == "hu":
      hubert_predict(signal)
    else:
      print("Unrecognised LID system. Exiting...")
      break

  config_file.close()