from transformers import Data2VecAudioForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h")
model = Data2VecAudioForSequenceClassification.from_pretrained("facebook/data2vec-audio-base-960h")

feature_extractor.save_pretrained("../downloaded_models/facebook/data2vec/fine-tuned/data2vec-audio-base-960h-extractor")
model.save_pretrained("../downloaded_models/facebook/data2vec/fine-tuned/data2vec-audio-base-960h-model")