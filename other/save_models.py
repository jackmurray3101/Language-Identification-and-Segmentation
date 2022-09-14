from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

feature_extractor.save_pretrained("./downloaded_models/superb_w2v_base_sid_extractor")
model.save_pretrained("./downloaded_models/superb_w2v_base_sid_model")