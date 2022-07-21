import os

languages = ['en', 'de', 'zh']
f = open("subset_labels.txt", "w")

for language in languages:
  dir_name = language + "_subset"
  files = os.listdir(dir_name)
  for file in files:
    f.write(f"{file} {language}\n")
f.close()