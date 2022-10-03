from transformers import Data2VecAudioForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base")
model = Data2VecAudioForSequenceClassification.from_pretrained("facebook/data2vec-audio-base")

feature_extractor.save_pretrained("../downloaded_models/facebook-data2vec-audio-base-extractor")
model.save_pretrained("../downloaded_models/facebook-data2vec-audio-base-model")