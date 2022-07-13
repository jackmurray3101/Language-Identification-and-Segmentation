import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment


tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
  print('Start speaking')
  while True:
    audio = r.listen(source) #pyaudio object
    data = io.BytesIO(audio.get_wav_data()) #list of bytes
    clip = AudioSegment.from_file(data) #numpy array
    x = torch.FloatTensor(clip.get_array_of_samples()) #tensor

    inputs = tokenizer(x, sample_rate=16000, return_tensors='pt', padding='longest').input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=.1)
    text = tokenizer.batch_decode(tokens) #tokens into a string

    print('You said: ', text)
