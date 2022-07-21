from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn
import librosa

sampling_rate = 16000

# Load waveforms
speech1, _ = librosa.load("./english.wav", sr=sampling_rate, mono=True)
speech2, _ = librosa.load("./mandarin.wav", sr=sampling_rate, mono=True)
speech = [speech1, speech2]


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
print(model)

# normalise inputs, pad or truncate, return attention vector
input_values = feature_extractor(speech,
                                 sampling_rate = sampling_rate,
                                 padding = 'max_length',
                                 max_length = sampling_rate * 20,
                                 return_tensors = 'pt',
                                 return_attention_mask = True,
                                 truncation = True)
print(input_values)


logits = model(**input_values).logits
print(logits)
print(logits.size())

model.classifier = nn.Linear(256, 3)

for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False
    

for param in model.wav2vec2.encoder.layers[10].parameters():
    param.requires_grad = True