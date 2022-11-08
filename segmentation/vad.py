from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")
output = pipeline("c:\\Users\\Jack\\Desktop\\Thesis\\code\\segmentation\\cleaned_data\\lan_14\\en-female.wav")

for speech in output.get_timeline().support():
  print(speech)