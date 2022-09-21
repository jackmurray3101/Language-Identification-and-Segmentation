import json
config_file = open("config1.json")
config = json.load(config_file)


languages = config["languages"]
print(languages[2])

labels_map = {}
for i in range(0, len(languages)):
  labels_map[languages[i]] = i


print(labels_map['el'])
label = 'de'
if label in languages:
  print('in')
else:
  print('out')