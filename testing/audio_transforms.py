from transformers import Wav2Vec2FeatureExtractor

class Transform:
  def __init__(self) -> None:
    pass
  
  def __call__(self, x):
    self.set_state()
    x = self.transform(x)
    return x
  
  def set_state(self):
    pass
  
  def transform(self, x):
    return self.do_transform(x)
  
  def do_transform(self,x):
    raise NotImplementedError
  
  
class Extractor(Transform):
  
  def __init__(self, base_name, max_length, sampling_rate):
    self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_name)
    self.sampling_rate = sampling_rate
    self.max_length = max_length
    
  def do_transform(self, x):
    
    features = self.extractor(x, sampling_rate=self.sampling_rate, padding='max_length', max_length=int(self.sampling_rate * self.max_length), return_tensors='pt', return_attention_mask=True, truncation=True)
    # (1,16000) -> (16000)
    return features['input_values'].reshape((-1)),features['attention_mask'].reshape((-1))

  