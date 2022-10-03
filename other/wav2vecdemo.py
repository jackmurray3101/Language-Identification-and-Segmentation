from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Data2VecAudioForSequenceClassification
import torch.nn as nn
import librosa

sampling_rate = 16000

# Load waveforms
speech1, _ = librosa.load("./english.wav", sr=sampling_rate, mono=True)
speech2, _ = librosa.load("./mandarin.wav", sr=sampling_rate, mono=True)
speech = [speech1, speech2]


# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
#model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53")


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base")
model = Data2VecAudioForSequenceClassification.from_pretrained("facebook/data2vec-audio-base")
#model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("../downloaded_models/superb/wav2vec2-base-sid-feature-extractor")
#model = Wav2Vec2ForSequenceClassification.from_pretrained("../downloaded_models/superb/wav2vec2-base-sid-model")
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

for param in model.data2vec_audio.feature_extractor.parameters():
    param.requires_grad = False
    

for param in model.data2vec_audio.encoder.layers[10].parameters():
    param.requires_grad = True